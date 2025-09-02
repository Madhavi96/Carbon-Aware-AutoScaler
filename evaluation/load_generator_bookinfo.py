from locust import HttpUser, TaskSet, task, constant
from locust import LoadTestShape, stats
import random
from random import randint, choice
import base64
# from atomic_queries import _query_advanced_ticket, _login
import time
# from query_advanced_ticket import kk
import json
from typing import List
import random
from typing import List
import string
import logging
import random
import time
from datetime import datetime, timedelta

stats.PERCENTILES_TO_REPORT = [0, 0.25, 0.5, 0.66, 0.75, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 0.9999, 1.0]

class BookInfoUserTasks(TaskSet):
    def __init__(self, parent):
        super().__init__(parent)
        

    @task(1)
    def get_productpage(self):
        self.client.get("/productpage")
    """
    # It seems productpage endpoint calls all other endpoints, so we can just call it
    # https://github.com/digitalocean/kubernetes-sample-apps/blob/master/bookinfo-example/README.md#overview
    # cannot invoke remianing endpoints directly, as they are not exposed as node ports
      
    @task(1)
    def get_details(self):
        self.client.get("/details")

    @task(1)
    def get_reviews(self):
        self.client.get("/reviews")
    
    @task(1)
    def get_ratings(self):
        self.client.get("/ratings")
    """


class WebsiteUser(HttpUser):
    def on_start(self):
        return super().on_start()

    def on_stop(self):
        return super().on_stop()
    host = "http://localhost:30012"
    wait_time = constant(1)
    tasks = [BookInfoUserTasks]

    
class StagesShape(LoadTestShape):
    """
    A simply load test shape class that has different user and spawn_rate at
    different stages.

    Keyword arguments:

        stages -- A list of dicts, each representing a stage with the following keys:
            duration -- When this many seconds pass the test is advanced to the next stage
            users -- Total user count
            spawn_rate -- Number of users to start/stop per second
            stop -- A boolean that can stop that test at a specific stage

        stop_at_end -- Can be set to stop once all stages have run.
    """

    def __init__(self):
        super().__init__()
        lines = []
        with open("random-1000max.req", 'r') as f:
        #with open("/ssj/ssj/train-ticket/train-ticket-auto-query2/sendflow/normalFlow.req", 'r') as f:
            lines = list(map(int, f.readlines()))
            lines = [x for i,x in enumerate(lines) if i%1==0]#在原来的基础上扩大了4倍
            self.lines = ([1]*5+lines+[1]*5)#又给了几个波谷
            #self.lines = lines#又给了几个波谷
    
    def tick(self):
        run_time = self.get_run_time()
        # for i in range(1, 100):
        #     return (i,1)
        #while True:
        for i, v in enumerate(self.lines):
            if run_time < (i+1)*5:# The interval is 5s
                tick_data = (v, 100)                
        # user_count -- Total user count
        # spawn_rate -- Number of users to start/stop per second when changing number of users
                # tick_data = (26, 100)
                return tick_data
        # for stage in self.stages:
        #     if run_time < stage["duration"]:
        #         tick_data = (stage["users"], stage["spawn_rate"])
        #         return tick_data

 
# if __name__ == '__main__':
#     lines = []
#     with open("/home/meng/random-100max.req", 'r') as f:
#         lines = list(map(int, f.readlines()))
#         lines = [x for i,x in enumerate(lines) if i%3==0]
#     print(len(lines))