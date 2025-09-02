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

class BoutiqueUserTasks(TaskSet):

    def __init__(self, parent):
        super().__init__(parent)
        self.products = [
            '0PUK6V6EV0',
            '1YMWWN1N4O',
            '2ZYFJ3GM2N',
            '66VCHSJNUP',
            '6E92ZMYYFZ',
            '9SIQT8TOJO',
            'L9ECAV7KIM',
            'LS4PSXUNUM',
            'OLJCESPC7Z']
    #
    @task(1)
    def index(self):
        self.client.get("/")
    #
    @task(2)
    def setCurrency(self):
        currencies = ['EUR', 'USD', 'JPY', 'CAD']
        self.client.post("/setCurrency",
            {'currency_code': random.choice(currencies)})
    @task(10)
    def browseProduct(self):
        self.client.get("/product/" + random.choice(self.products))

    @task(2)
    def viewCart(self):
        self.client.get("/cart")


    """
    These don´t work

    @task(3)
    def addToCart(self):
        product = random.choice(self.products)
        self.client.get("/product/" + product)
        self.client.post("/cart", {
            'product_id': product,
            'quantity': random.choice([1,2,3,4,5,10])})

    @task(1)
    def checkout(self):
        self.addToCart()
        self.client.post("/cart/checkout", {
            'email': 'someone@example.com',
            'street_address': '1600 Amphitheatre Parkway',
            'zip_code': '94043',
            'city': 'Mountain View',
            'state': 'CA',
            'country': 'United States',
            'credit_card_number': '4432-8015-6152-0454',
            'credit_card_expiration_month': '1',
            'credit_card_expiration_year': '2039',
            'credit_card_cvv': '672',
        })
    """

    @task(3)
    def addToCart(self):
        product = random.choice(self.products)
        self.client.get(f"/product/{product}")
        self.client.post(
            "/cart",
            data={
                "product_id": product,
                "quantity": str(random.choice([1, 2, 3, 4, 5, 10]))
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

    @task(1)
    def checkout(self):
        # Add at least one product to cart first
        self.addToCart()
        self.client.post(
            "/cart/checkout",
            data={
                "email": "someone@example.com",
                "street_address": "1600 Amphitheatre Parkway",
                "zip_code": "94043",
                "city": "Mountain View",
                "state": "CA",
                "country": "United States",
                "credit_card_number": "4432-8015-6152-0454",
                "credit_card_expiration_month": "1",
                "credit_card_expiration_year": "2039",
                "credit_card_cvv": "672",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

class WebsiteUser(HttpUser):
    def on_start(self):
        return super().on_start()

    def on_stop(self):
        return super().on_stop()
    # host = "http://localhost:31642"
    wait_time = constant(1)
    tasks = [BoutiqueUserTasks]

    
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
        with open("random-100max.req", 'r') as f:
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