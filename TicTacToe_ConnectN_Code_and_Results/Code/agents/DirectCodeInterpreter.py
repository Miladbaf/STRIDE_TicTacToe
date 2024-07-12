import sys
import os
import re
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

import random
from copy import deepcopy
from dotenv import load_dotenv
load_dotenv()
import instructor
from openai import OpenAI
from typing import List
from pydantic import BaseModel, create_model
import tenacity

import os
from dotenv import load_dotenv
import openai
import logging  
import time 

from typing_extensions import override
from openai import AssistantEventHandler

types = {
    'string': str,
    'integer': int,
    'boolean': bool,
    'float': float,
    'List[str]': List[str],
}

def initialize_openai_client():
    """
    Initializes the OpenAI client by loading the API key from environment variables.
    """
    load_dotenv()  
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    openai.api_key = api_key
    return openai  

class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)
        
    @override
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)
        
    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)
    
    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)

class DirectCodeInterpreterAgent:
    def __init__(self, problem_description, logger, demo=None, read_file=False, engine="gpt-3.5-turbo") -> None:
        self.logger = logger
        self.engine = engine
        self.read_file = read_file
        # Initialize the OpenAI client
        self.client = initialize_openai_client()
        # Create a new assistant
        if demo is None:
           instructions = "You are a world class intelligent agent capable of solving various classes of decision making problems. For each decision making problem you encounter next, you will be given the description of the problem setup and your objective. Your need to carefully reason about the problem step-by-step, and make optimal decisions for the encountered problem instance. You are provided with a code interpreter, and you should write and run code to answer the questions."
        else:
           instructions = "You are a world class intelligent agent capable of solving various classes of decision making problems. For each decision making problem you encounter next, you will be given the description of the problem setup and your objective. Your need to carefully reason about the problem, and make optimal decisions for the encountered problem instance. You are provided with a code interpreter, and you should write and run code to answer the questions.\nHere is an example:\n{}".format(demo)
        
        self.assistant = self.client.beta.assistants.create(
            name = "Strategic Decision Maker",
            instructions = instructions,
            tools = [{"type": "code_interpreter"}], # The tools that the assistant can use, her it is the code interpreter tool, could be also file interpreter etc.
            model = self.engine # The newes model that support assistants
        )
        self.problem_description = problem_description
        self.mover_client = instructor.from_openai(OpenAI())
        self.reset()

    def reset(self):
        self.thread = self.client.beta.threads.create()
        logging.info(f'Thread Object: {self.thread}')  # DEBUG: Info about the thread object

    def get_instance_info(self, instance_descript):
        self.instance_descript = deepcopy(instance_descript+"\nYou can access your working memory in the working_memory.json file.")
        # self.messages.append({"role":"assistant", "content":instance_descript})
        self.file = self.client.files.create(
        file = open("agents/working_memory.json", "rb"),
        purpose='assistants'
        )
    
    def update(self, state, reward):
        pass

    def reason(self, question):
        if self.read_file:
            message = self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=self.problem_description+"\n"+self.instance_descript+"\n"+question,
                attachments=[
                    {
                    "file_id": self.file.id,
                    "tools": [{"type": "code_interpreter"}]
                    }
                ]
            )
        else:
            message = self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=self.problem_description+"\n"+self.instance_descript+"\n"+question
            )
        with self.client.beta.threads.runs.stream(
        thread_id=self.thread.id,
        assistant_id=self.assistant.id,
        # instructions="Please address the user as Jane Doe. The user has a premium account.",
        event_handler=EventHandler(),
        ) as stream:
            stream.until_done()

    def extract_move_from_message(self, message):
        pattern = re.compile(
        r"mark in position\s*(\d+)|"
        r"best move.*?position\s*(\d+)|"
        r"column\s*(\d+)", 
        re.IGNORECASE
        )
        match = pattern.search(message)
        if match:
            for group in match.groups():
                if group:
                    return int(group)
        raise ValueError("No valid move found in the message")

    def move(self, state):
        # self.reason(state.textual_descript)
        # self.messages.append({"role":"user", "content":state.textual_descript})

        message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=state.textual_descript+"\n"+"Provide a single integer, i.e., the index of the action, as your answer"
        )

        with self.client.beta.threads.runs.stream(
        thread_id=self.thread.id,
        assistant_id=self.assistant.id,
        instructions="Please try to be concise in your response.",
        event_handler=EventHandler(),
        ) as stream:
            stream.until_done()

        # Retrive all the messages which are inside run
        messages = self.client.beta.threads.messages.list(
            thread_id = self.thread.id,
        )

        result = messages.data[0].content[0].text.value
        
        Action = create_model(
            'Action',
            **{
                property_name: (types[property_type], description)
                for property_name, property_type, description in state.action_schema
            },
            __base__=BaseModel,
        )

        try:
            action = self.mover_client.chat.completions.create(
                model=self.engine,
                temperature=0,
                response_model=Action,
                messages=[{"role":"system", "content":"Select an action based on the reasoning provided below."}, {"role":"assistant", "content":result}],
                max_tokens=1000,
                max_retries=tenacity.Retrying(
                    stop=tenacity.stop_after_attempt(3),
                ),
            )
            if state.is_valid_action(action.action):
                return action.action
            else:
                # self.messages.append({"role":"assistant", "content":"Failed to get correct action format. Will take a random action instead."})
                self.logger.write("Failed to get correct action format. Will take a random action instead.")
                if state.actions == [0.0, 1.0]:
                    # offer space of bargaining
                    return random.uniform(0, 1)
                else:
                    return random.choice(state.actions)
        except:
            # self.messages.append({"role":"assistant", "content":"Failed to get correct action format. Will take a random action instead."})
            self.logger.write("Failed to get correct action format. Will take a random action instead.")
            if state.actions == [0.0, 1.0]:
                # offer space of bargaining
                return random.uniform(0, 1)
            else:
                random.choice(state.actions)

    def move_connect(self, state):
        # Present the current board and ask for the best move
        board_description = f"Current board state:\n{state.textual_descript}\nPlease provide the best move."
        message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=board_description
        )
        with self.client.beta.threads.runs.stream(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            event_handler=EventHandler(),
        ) as stream:
            stream.until_done()
        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread.id,
        )

        messages_list = list(messages.data)
        last_relevant_message = self.find_relevant_message(messages_list)

        if last_relevant_message:
            try:
                action = self.extract_move_from_message(last_relevant_message)  # pass the string directly
                return action
            except ValueError as e:
                self.logger.write(f"Failed to extract move: {str(e)}")
                return None
        else:
            self.logger.write("No relevant message found for extracting the move")
            return None
    
    def find_relevant_message(self, messages):
        for message in reversed(messages):
            if message.role == 'assistant':
                text_content = ' '.join(block.text.value for block in message.content if hasattr(block, 'text') and hasattr(block.text, 'value'))

                patterns = [
                    r"is to place their mark in position (\d+)",
                    r"is to place the mark on position (\d+)",
                    r"best move.*?position (\d+)",
                    r"column (\d+)"
                ]
                for pattern in patterns:
                    if re.search(pattern, text_content, re.IGNORECASE):
                        return text_content
        
        return None