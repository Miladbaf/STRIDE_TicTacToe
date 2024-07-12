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
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, NonNegativeInt, NonNegativeFloat, create_model, model_validator
import tenacity

# import the tools for all the decision making problems
from envs.tabular_mdp.tools import *
from envs.bargain_alternate_singleissue.tools import *
from envs.connectn.tools import *
from envs.tictactoe.tools import *

types = {
    'string': str,
    'integer': int,
    'boolean': bool,
    'float': float,
    'List[str]': List[str],
}

class StriDeAgent:
    def __init__(self, problem_description, demo, tool_names, init_memory, logger, engine="gpt-3.5-turbo", llm_validator=False, game=None) -> None:
        self.working_memory = init_memory
        self.game = game
        self.logger = logger
        self.client = instructor.from_openai(OpenAI())
        self.engine = engine
        self.llm_validator = llm_validator
        self.initial_messages = [
            {
                "role":"system", 
                "content":"You are a world class intelligent agent capable of solving various classes of decision making problems. For each decision making problem you encounter next, you will be given the description of the problem setup and your objective. Your need to carefully reason about the problem step-by-step, and make optimal decisions for the encountered problem instance. You are provided with a set of tools and examples showing how to use these tools to solve this problem."
             },
             {
                 "role":"assistant",
                 "content":problem_description
             }
        ]
        self.initial_messages += demo
        self.messages = deepcopy(self.initial_messages)
        self.tool_names = tool_names
        self.instance_descript = ""

    def reset(self):
        self.messages = deepcopy(self.initial_messages)
        if "V" in self.working_memory and "Q" in self.working_memory:
            # reset V and Q
            self.working_memory["V"] = np.zeros((self.working_memory["epLen"],self.working_memory["nState"]))
            self.working_memory["Q"] = np.zeros((self.working_memory["epLen"],self.working_memory["nState"],self.working_memory["nAction"]))
        else:
            self.working_memory.memo.clear()
            self.working_memory.depth_scores.clear()

    def get_instance_info(self, instance_descript):
        self.instance_descript = deepcopy(instance_descript)
        self.messages.append({"role":"assistant", "content":instance_descript})

    def update(self, state, reward):
        pass

    def move(self, state):
        if self.game == 'TicTacToe' or self.game == 'ConnectN':    
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
                response = self.client.chat.completions.create(
                    model=self.engine,
                    temperature=0,
                    response_model=Action,
                    messages=self.messages[-1:],  # using the last Thought
                    max_tokens=1000,
                )

                response_text = getattr(response, 'text', str(response)) 
                match = re.search(r"action=(\d+)", response_text)
                if match:
                    action = int(match.group(1))
                    if state.is_valid_action(action):
                        return action
                    else:
                        raise ValueError("Invalid action selected")
                else:
                    raise ValueError("No valid action found in the response")
            except Exception as e:
                self.logger.write(f"Error retrieving or validating action: {str(e)}")
                fallback_action = random.choice(state.actions) if state.actions != [0.0, 1.0] else random.uniform(0, 1)
                return fallback_action
        else:
            # self.messages.append({"role":"user", "content":state.textual_descript})
            self.reason(state.textual_descript)
            Action = create_model(
                'Action',
                **{
                    property_name: (types[property_type], description)
                    for property_name, property_type, description in state.action_schema
                },
                __base__=BaseModel,
            )

            # self.reason("Which action I should take?")

            try:
                action = self.client.chat.completions.create(
                    model=self.engine,
                    temperature=0,
                    response_model=Action,
                    messages=self.messages[-1:], # only need the last Thought of the reasoning process
                    max_tokens=1000,
                    max_retries=tenacity.Retrying(
                        stop=tenacity.stop_after_attempt(3),
                    ),
                )
                print(f"\n\nACTION IS: {action.actions}")
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
                    return random.choice(state.actions)

    def reason(self, question):
        # print("===")
        # print(self.messages)
        tool_names = self.tool_names
        # a string enum for all available tools that agent can choose from
        ToolName = StrEnum('ToolName', tool_names)

        if self.llm_validator:
            class Validation(BaseModel):
                is_valid: bool = Field(
                    ..., description="Whether the value is valid given the rules"
                )
                error_message: Optional[str] = Field(
                    ...,
                    description="The error message if the value is not valid, to be used for re-asking the model",
                )
            def validator(values):
                ms = [
                    {
                        "role": "system",
                        "content": "You are a validator for a reasoner, whose job is to make sure the reasoner is on the right track to answer the question. For each question, the reasoner will generate a reasoning process as a sequence of Thoughts, and you need to compare it with the reasoning process provided in the example that answers the same question. Determine whether reasoner's newly generated Thought is valid. If not, why.",
                    },
                    {
                        "role": "assistant",
                        "content": "Here are some examples:\n"
                    },
                ]+self.initial_messages[1:]+[
                    {
                        "role": "assistant",
                        "content": "Now we are facing a new problem instance:\n{}".format(self.instance_descript)
                    },
                    {
                        "role": "assistant",
                        "content": "Here is the question the reasoner is trying to answer now:\n[Reasoner]\n"
                    }
                ]+self.messages_current_question+[
                    {
                        "role":"assistant",
                        "content":"\nNow by comparing with Thoughts provided in the example that answers the same question, judge whether the following Thought (which is the next step of reasoning) is valid.".format(tool_names)
                    },
                    {
                        "role": "user",
                        "content": "Thought: {}\n list of operations: {}\nExit: {}".format(values["text"],values["operations"],values["exit"])
                    },
                ]
                resp = self.client.chat.completions.create(
                    model=self.engine,
                    messages=ms,
                    response_model=Validation,
                )
                if not resp.is_valid:
                    self.logger.write("==== validator ====")
                    self.logger.write(resp.error_message)
                    # self.logger.write("====    mid    ====")
                    # self.logger.write(ms)
                    # self.logger.write(values["text"])
                    self.logger.write("====   end     ====")
                    raise ValueError(resp.error_message)
                return values
        else:
            def validator(values):
                if values["operations"] != []:
                    for op in values["operations"]:
                        if op not in tool_names:
                            self.logger.write("==== validator ====")
                            self.logger.write("You can only choose from {}. Only the name of the operation is needed. Do not add the inputs!""".format(tool_names))
                            self.logger.write("====   end     ====")
                            raise ValueError("You can only choose from {}. Only the name of the operation is needed. Do not add the inputs!""".format(tool_names))
                if values["operations"] != [] and values["exit"]:
                    self.logger.write("==== validator ====")
                    self.logger.write("Conflict detected: By setting Exit to True, you won't be able to execute any of the selected operation. Now decide, whether you want to continue reasoning, or exit. For valua iteration, only exit after finishing time step 0!")
                    self.logger.write("====   end     ====")
                    raise ValueError("Conflict detected: By setting Exit to True, you won't be able to execute any of the selected operation. Now decide, whether you want to continue reasoning, or exit. For valua iteration, only exit after finishing time step 0!")
                return values

        class Thought(BaseModel):
            text: str = Field(
                ...,
                description="a textual description of the current step of reasoning"
            )
            exit: bool = Field(
                ...,
                description="set to True, if it is time to exit reasoning process",
                # """Set exit to True only when you have finished the whole reasoning process and made a decision about the answer. Be very careful not to exit before you have finished reasoning!""",
            )
            operations: List[ToolName] = Field(
                ...,
                description="""The list of operations to be executed for the current step of reasoning. You can only choose from {}. Only the name of the operation is needed. Do not add the inputs!""".format(tool_names),
            )

            # @model_validator(mode="before")
            # @classmethod
            # def chain_of_thought_makes_sense(cls, data):
            #     return validator(data)

            @field_validator("operations")
            @classmethod
            def operations_is_valid(cls, v: List[ToolName]):
                if v != []:
                    for op_name in v:
                        assert op_name.name in tool_names, "must pick from the available operations: {}. And be very careful not to make any spelling error or use the incorrect case of the letter.".format(op_name, tool_names)
                return v
        
        self.messages_current_question = []
        self.messages.append({"role":"user", "content":"Question: "+question})
        self.messages_current_question.append({"role":"user", "content":"Question: "+question})
        self.logger.write("Question: "+question)

        # print("===========")
        # print(self.messages)

        try:
            thought: Thought = self.client.chat.completions.create(
                model=self.engine,
                temperature=0,
                response_model=Thought,
                messages=self.messages,
                max_tokens=1000,
                max_retries=tenacity.Retrying(
                    stop=tenacity.stop_after_attempt(4),
                ),
            )
            self.messages.append({"role":"assistant", "content":"Thought: "+thought.text})
            self.messages_current_question.append({"role":"assistant", "content":"Thought: "+thought.text})
            self.logger.write("Thought: "+thought.text)
        except Exception as e:
            err = "exit reasoning process due to failed thought generation, with the following error\n"+getattr(e, 'message', repr(e))
            self.logger.write(err)
            # self.messages.append(err)
            return

        while not thought.exit:
            op_name_ls = thought.operations
            if op_name_ls == []:
                break
            for op_name in op_name_ls:
                op = self.client.chat.completions.create(
                    model=self.engine,
                    temperature=0,
                    response_model=eval(op_name.name),
                    messages=[{"role":"system", "content":"Based on the description of the problem instance and the Thought, output the operation to be taken."},{"role":"assistant", "content":self.instance_descript+"\nThought: "+thought.text}],
                    max_tokens=1000,
                    max_retries=tenacity.Retrying(
                        stop=tenacity.stop_after_attempt(3),
                    ),
                )
                op_text = "Operation: call function {} with inputs {}".format(op_name.name, op)
                self.logger.write(op_text)
                self.messages.append({"role":"assistant", "content":op_text})
                self.messages_current_question.append({"role":"assistant", "content":op_text})
                try:
                    res_text = op.execute(game=self.game, working_memory=self.working_memory)
                    self.logger.write("Result: {}".format(res_text))
                    self.messages.append({"role":"assistant", "content":"Result: {}".format(res_text)})
                except Exception as e:
                    res_text = "Result: execution of {} failed, with the following error\n".format(op_name.name)+getattr(e, 'message', repr(e))
                    self.logger.write(res_text)
                    # self.messages.append(res_text)
            # generate next thought
            try:
                thought: Thought = self.client.chat.completions.create(
                    model=self.engine,
                    temperature=0,
                    response_model=Thought,
                    messages=self.messages,
                    max_tokens=1000,
                    max_retries=tenacity.Retrying(
                        stop=tenacity.stop_after_attempt(4),
                    ),
                )
                self.messages.append({"role":"assistant", "content":"Thought: "+thought.text})
                self.messages_current_question.append({"role":"assistant", "content":"Thought: "+thought.text})
                self.logger.write("Thought: "+thought.text)
            except Exception as e:
                err = "exit reasoning process due to failed thought generation, with the following error\n"+getattr(e, 'message', repr(e))
                self.logger.write(err)
                # self.messages.append(err)
                break