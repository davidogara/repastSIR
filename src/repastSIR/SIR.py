import os
from tempfile import NamedTemporaryFile
import argparse
import json
from mpi4py import MPI
from dataclasses import dataclass
from typing import Dict, Set, List
import math
from collections import Counter
from repast4py import core, space, schedule, logging, random
from repast4py import context as ctx
from repast4py.space import DiscretePoint, ContinuousPoint
from repast4py.space import BorderType, OccupancyType
import numpy as np
from numba import int32
from numba.experimental import jitclass    
from time import time
import pandas as pd
from copy import copy
import yaml
from joblib import Parallel, delayed
spec = [    
    ('mo', int32[:]),   
    ('no', int32[:]),
    ('xmin', int32),    
    ('ymin', int32),
    ('ymax', int32),
    ('xmax', int32)
]
model = None

@jitclass(spec)   
class GridNghFinder:
    '''
    Neighborhood finder, copied from: https://github.com/Repast/repast4py/blob/master/examples/zombies/zombies.py#L39
    '''

    def __init__(self, xmin, ymin, xmax, ymax):   
        self.mo = np.array([0,-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)    
        self.no = np.array([0,1, 1, 1, 0, 0, -1, -1, -1], dtype=np.int32)
        self.xmin = xmin    
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def find(self, x, y):    
        xs = self.mo + x    
        ys = self.no + y    

        xd = (xs >= self.xmin) & (xs <=self.xmax)    
        xs = xs[xd]   
        ys = ys[xd]

        yd = (ys >= self.ymin) & (ys <= self.ymax)    
        xs = xs[yd]
        ys = ys[yd]

        return np.stack((xs, ys, np.zeros(len(ys), dtype=np.int32)), axis=-1)

class Agent(core.Agent):
    r'''
    Agent in the SIR model: agents have:
        * a pt (location in x,y grid space)
        * a state (S,I,R)
        * a time_infected (number of timesteps infected/infectious for)
    '''
    TYPE = 0 

    def __init__(self, local_id: int, rank: int, pt: DiscretePoint):
        super().__init__(id=local_id, type=Agent.TYPE, rank=rank)    
        self.pt = pt
        self.pt0 = pt # initial pt
        self.state = "S"
        self.time_infected = 0

    def move(self,grid):
        '''
        agents move randomly
        
        Parameters
        ----------
        grid: the 2D model grid
        '''

        xnew = random.default_rng.choice([-1,0,1])
        ynew = random.default_rng.choice([-1,0,1])
        pt = np.array([self.pt.x + xnew, self.pt.y + ynew,0])
        at = DiscretePoint(0, 0, 0)
        at._reset_from_array(pt)
        new_pt = grid.move(self,at)
        if new_pt is not None:
            self.pt = new_pt

        
        

    def infect(self, grid):
        '''
        Infect Susceptible neighbors with probability pars["beta"]

        Parameters
        ----------
        grid: the 2D model grid
        
        '''
        random_draws = random.default_rng.uniform(size=1000)
        counter = 0
        if self.state == "I":
            pt = grid.get_location(self)
            nghs = model.ngh_finder.find(pt.x, pt.y)
            at = DiscretePoint(0, 0, 0)
            
            for ngh in nghs:
                at._reset_from_array(ngh)
                for agent in grid.get_agents(at):
                    if agent.state == "S":
                        if random_draws[counter] < model.pars['beta']:
                            agent.get_infected()
                    counter +=1
            self.time_infected +=1

    def get_infected(self):
        r'''
        Change state to infected and set time_infected to 0
        '''
        self.state = "I"
        self.time_infected = 0

    def recover(self):
        '''
        Change state to Recovered after "gamma" time being infected
        '''
        if self.state == "I" and self.time_infected >= model.pars['gamma']:
            self.state = "R"


    def save(self):
        return self.uid, self.state, self.time_infected, self.pt.coordinates

agent_cache = {}
def restore_agent(agent_data):
    '''
    restore agents across processes (not used here since model is single-core)
    
    Parameters
    ----------
    agent_data: tuple of agent data containing uid, state, time_infected
    '''
    uid = agent_data[0]
    state = agent_data[1]
    time_infected = agent_data[2]
    pt_array = agent_data[3]
    pt = DiscretePoint(pt_array[0],pt_array[1])
    if uid in agent_cache:
        agent = agent_cache[uid]
    else:
        agent = Agent(uid[0],uid[1],uid[2])
        agent.state = state
        agent.time_infected = time_infected
        agent.pt = pt
    return agent


@dataclass
class Counts:
    S: int = 0
    I: int = 0
    R: int = 0
class Model:
    def __init__(self,comm,pars):
        self.comm = comm
        self.pars = pars
        self.context = ctx.SharedContext(comm)
        self.rank = self.comm.Get_rank()
        if self.pars['override_seed']:
            random.init(pars['override_seed'])
        else:    
            random.init(pars['seed'])
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(pars['stop.at'])
        self.runner.schedule_end_event(self.at_end)
        box = space.BoundingBox(pars['world.xmin'], pars['world.xmax'], pars['world.ymin'], pars['world.ymax'], 0, 0)
        self.box = box
        self.grid = space.SharedGrid('grid', bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple,
                                     buffer_size=2, comm=comm)
        self.context.add_projection(self.grid)
        self.ngh_finder = GridNghFinder(box.xmin, box.ymin, box.xextent, box.yextent)
        self.counts = Counts()
        if pars['return_data']:
            self.data_set = [{'tick':0,'S':pars['n'],'I':0,'R':0}]
        else:
            loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
            self.data_set = logging.ReducingDataSet(loggers, self.comm, pars['counts_file'])

        self.infected: Set[Agent] = set()
        self.init_agents()

    def assign_location_of_index_case(self):
        # choose location of index case
        if self.pars['index_case'] == 'index_case_based_on_seed_odd_even':
            bounds = self.grid.get_local_bounds()
            if (self.pars['seed'] % 2) == 0:
                # for even seeds, place index in center
                xc = int((bounds[1] - bounds[0]) / 2)
                yc = int((bounds[3] - bounds[2]) / 2)
                
                x,y = xc, yc
            else:
                # for odd seeds, place index on top right corner
                x,y = bounds.xextent, bounds.yextent
        if self.pars['index_case'] == 'index_case_seed_size':
            bounds = self.grid.get_local_bounds()
            # center
            xc = int((bounds[1] - bounds[0]) / 2)
            yc = int((bounds[3] - bounds[2]) / 2)
            # get all points within radius of center (using inf norm like a box)
            x,y = xc + self.pars['seed'], yc + self.pars['seed']
            #xpts = np.arange(box.xmin,box.xextent+1)
            #ypts = np.arange(box.ymin,box.xextent+1)
            #pts = np.array([(x,y) for x in xpts for y in ypts])
            #dist = np.linalg.norm(pts - np.array([xc,yc]),ord=np.inf,axis=1)
            
            #pts_eligible = pts[dist==self.pars['seed']]
            #x, y = rng.choice(pts_eligible,size=1).squeeze()
        else:
            x,y = self.pars['init_infect_loc']
        return x,y

    def init_agents(self):
        rng = random.default_rng
        n = self.pars['n']
        # create a set of agents on main diagonal from center to top right corner
        
        bounds = self.grid.get_local_bounds()
        # center
        xc = int((bounds[1] - bounds[0]) / 2)
        yc = int((bounds[3] - bounds[2]) / 2)
        agents_on_diagonal = np.arange(xc,self.box.xextent+1)
        
        for aid, i in enumerate(agents_on_diagonal):
            pt = DiscretePoint(i,i,0)
            a = Agent(aid,self.rank,pt = pt)
            self.context.add(a)
            new_pt = self.grid.move(a, pt)
        x,y = self.assign_location_of_index_case()
        # one of the above instantiated agents is the index case
        pt = DiscretePoint(x,y,0)
        a = self.grid.get_agent(pt)
        a.state = "I"
        self.infected.add(a)
        
        
        # create N - 1 remaining agents
        offset = aid + 1
        for aid, i in enumerate(range(len(agents_on_diagonal),n)):
            pt = self.grid.get_random_local_pt(rng)
            a = Agent(offset + aid,self.rank,pt = pt)
            self.context.add(a)
            new_pt = self.grid.move(a, pt)
            a.pt = new_pt
    
        if self.pars['save_movement']:
            out_file = f'./agent_movement/override_seed_{self.pars["override_seed"]}_run_number_{self.pars["run_number"]}.csv'
            agents = self.context.agents()
            agent_df = []
            for a in agents:
                pt = a.pt
                agent_info = {'uid':a.uid[0],'pt_x':pt.x,'pt_y':pt.y,'state':a.state}
                agent_df.append(agent_info)
            agent_df = pd.DataFrame(agent_df).sort_values('uid')
            agent_df['tick'] = 0
            agent_df.to_csv(out_file,index=False)

    def at_end(self):
        if self.pars['return_data']:
            pass
        else:
            self.data_set.close()

    def step(self):
        r'''
        Run one step of SIR Model.

        Iterates over agents in the context following rule-based activation. Agents are shuffled at each timestep.

        This means that all agents move, then infect/recover
        
        Parameters
        ----------
        self

        Returns
        -------
        None
        '''
        tick = self.runner.schedule.tick
        for a in self.context.agents(shuffle=False):
            a.move(self.grid)

        for a in self.context.agents(shuffle=False):
            a.infect(self.grid)
            a.recover()
        if self.pars['save_movement']:
            out_file = f'./agent_movement/override_seed_{self.pars["override_seed"]}_run_number_{self.pars["run_number"]}.csv'
            agents = self.context.agents()
            agent_df = []
            for a in agents:
                pt = a.pt
                agent_info = {'uid':a.uid[0],'pt_x':pt.x,'pt_y':pt.y,'state':a.state}
                agent_df.append(agent_info)
            agent_df = pd.DataFrame(agent_df).sort_values('uid')
            agent_df['tick'] = tick
            agent_df.to_csv(out_file,index=False,mode='a',header=False)
        self.log_counts(tick)   
        #self.context.synchronize(restore_agent)
    def run(self):
        self.runner.execute()

    def log_counts(self, tick: int):
        r'''
        Log counts at each tick. Depending on pars["return_data"] type, either creates a Counter of agent state values or records S,I,R counts
        with a dataclass.

        Parameters
        ----------
        tick: model timestep

        Returns
        -------
        None
        '''
        states = [A.state for A in self.context.agents()]
        state_counts = Counter(states)
        self.counts.S = state_counts.get('S',0)
        self.counts.I = state_counts.get('I',0)
        self.counts.R = state_counts.get('R',0)
        if self.pars['return_data']:
            S = state_counts.get('S',0)
            I = state_counts.get('I',0)
            R = state_counts.get('R',0)
            self.data_set.append({'tick':tick,'S':S,'I':I,'R':R})
        else:
            self.data_set.log(tick) 

def run(params: Dict,y_true = None, return_err = False):
    r'''
    
    Creates and runs the SIR Model.

    Parameters
    ----------
    params: dict
        the model input parameters
    '''
    global model


    beta_range = np.array([0.05,0.10])
    beta = beta_range[0] + params[0] * (beta_range.max() - beta_range.min())
    seed = int(params[1])
    
    here = os.path.dirname(__file__)
    base_pars = read_yaml(os.path.join(here,'input','pars_box_norm_ground_truth.yaml'))
    params = base_pars.copy()
    params.update({'beta':beta,'seed':seed})


    model = Model(MPI.COMM_WORLD, params)
    
    model.run()
    if params['return_data'] and y_true is not None and return_err:
        sim = pd.DataFrame(model.data_set)
        
        ans = {'out': ((sim['I']-y_true)**2).sum(),
               'res': sim
            }
        return ans
    if params['return_data']:
        sim = pd.DataFrame(model.data_set)
        return sim

def generate_ground_truth():
    here = os.path.dirname(__file__)

    pars = read_yaml(os.path.join(here,'input','pars_box_norm_ground_truth.yaml'))
    return run(pars)



 
    
def run_batch_cmd():

    parser = argparse.ArgumentParser()
    parser.add_argument("--pars", type=str, default="{}", help="json parameters string")
    args = parser.parse_args()
    with open(args.pars) as stream:
        pars = json.load(stream)
    X = pd.DataFrame(pars)[['beta','seed']].values
    result = run_batch(X)
    # parse results
    result['Y'] = result['Y'].tolist()
    result['sims'] = result['sims'].to_dict(orient='records')
    result_json = json.dumps(result)
    print(result_json)
    
    
    


def run_batch(X,ytrue = None):
    if ytrue is None:
        here = os.path.dirname(__file__)
        fp = os.path.join(here,'input','repastSIR_ground_truth.csv')
        ground_truth = pd.read_csv(fp)
        ytrue = ground_truth['I'].values
    def parallel_func(par):
        simout = run_sim_err(par,ytrue=ytrue)
        return simout
    parallel = Parallel(n_jobs=-1,return_as="list")
    pars = [x for x in X]
    results =  parallel(delayed(parallel_func)(par) for par in pars)

    Y = []
    sims = []
    for i, par in enumerate(X):
        res = results[i]
        sim = res['res']
        sim['sim_id'] = i
        sims.append(sim)
        Y.append(res['out'])
    sims = pd.concat(sims)
    Y = np.array(Y)
    return {'Y':Y,'sims':sims}   

def run_sim_err(par,ytrue = None):
    return run(par,y_true=ytrue, return_err=True)


def read_yaml(path: str) -> dict:
    r'''
    Reads in a yaml file to a dict
    '''
    with open(path,'rb') as stream:
        return yaml.safe_load(stream)
    
if __name__ == "__main__":
    # a minimal example of the SIR parameters
    params = read_yaml('input/pars_box_norm_ground_truth.yaml')
    start = time()
    run(params)
    end = time()
    print(f"Time: {end-start:0.02f} seconds")