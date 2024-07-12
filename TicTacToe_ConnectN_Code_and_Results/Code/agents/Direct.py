import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

import random
from copy import deepcopy
from enum import Enum
from strenum import StrEnum
from dotenv import load_dotenv
load_dotenv()
import instructor
from openai import OpenAI
from typing import List
from pydantic import BaseModel, create_model
import tenacity

types = {
    'string': str,
    'integer': int,
    'boolean': bool,
    'float': float,
    'List[str]': List[str],
}

class DirectAgent:
    def __init__(self, problem_description, logger, engine="gpt-3.5-turbo") -> None:
        self.logger = logger
        self.client = instructor.from_openai(OpenAI())
        self.engine = engine
        self.initial_messages = [
            {
                "role":"system", 
                "content":"You are a world class intelligent agent capable of solving various classes of decision making problems. For each decision making problem you encounter next, you will be given the description of the problem setup and your objective. Your need to carefully reason about the problem step-by-step, and make optimal decisions for the encountered problem instance."
             },
             {
                 "role":"assistant",
                 "content":problem_description
             }
        ]
        self.messages = deepcopy(self.initial_messages)
        self.instance_descript = ""

    def reset(self):
        self.messages = deepcopy(self.initial_messages)

    def get_instance_info(self, instance_descript):
        self.instance_descript = deepcopy(instance_descript)
        self.messages.append({"role":"assistant", "content":instance_descript})

    def update(self, state, reward):
        pass

    def move(self, state):
        self.reason(state.textual_descript)
        Action = create_model(
            'Action',
            **{
                property_name: (types[property_type], description)
                for property_name, property_type, description in state.action_schema
            },
            __base__=BaseModel,
        )

        try:
            action = self.client.chat.completions.create(
                model=self.engine,
                temperature=0,
                response_model=Action,
                messages=self.messages,
                max_tokens=1000,
                max_retries=tenacity.Retrying(
                    stop=tenacity.stop_after_attempt(3),
                ),
            )
            if state.is_valid_action(action.action):
                return action.action
            else:
                self.messages.append({"role":"assistant", "content":"Failed to get correct action format. Will take a random action instead."})
                self.logger.write("Failed to get correct action format. Will take a random action instead.")
                if state.actions == [0.0, 1.0]:
                    # offer space of bargaining
                    return random.uniform(0, 1)
                else:
                    return random.choice(state.actions)
        except:
            self.messages.append({"role":"assistant", "content":"Failed to get correct action format. Will take a random action instead."})
            self.logger.write("Failed to get correct action format. Will take a random action instead.")
            if state.actions == [0.0, 1.0]:
                # offer space of bargaining
                return random.uniform(0, 1)
            else:
                random.choice(state.actions)
    
    def reason(self, question):
        # print("======")
        # print(self.messages)
        self.messages.append({"role":"user", "content":question})
        self.logger.write(question)
        response = self.client.chat.completions.create(
            model=self.engine,
            temperature=0,
            response_model=None,
            messages=self.messages,
            max_tokens=1000,
            max_retries=tenacity.Retrying(
                stop=tenacity.stop_after_attempt(3),
            ),
        )
        message = response.choices[0].message
        self.messages.append({"role":"assistant", "content":message.content})
        self.logger.write(message.content)