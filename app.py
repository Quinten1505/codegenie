from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class CAMELAgent:

    def __init__(
        self,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.system_message = system_message
        self.model = model
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages

    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
        messages = self.update_messages(input_message)

        output_message = self.model(messages)
        self.update_messages(output_message)

        return output_message

assistant_role_name = "Agent420"
user_role_name = "Agent69"
task = "Develop a simple calculator app in python"
word_limit = 50 # word limit for task brainstorming

task_specifier_sys_msg = SystemMessage(content="You can make a task more specific.")
task_specifier_prompt = (
"""Here is a task that {assistant_role_name} will help {user_role_name} to complete: {task}.
Please make it more specific. Be creative and imaginative.
Please reply with the specified task in {word_limit} words or less. Do not add anything else."""
)
task_specifier_template = HumanMessagePromptTemplate.from_template(template=task_specifier_prompt)
task_specify_agent = CAMELAgent(task_specifier_sys_msg, ChatOpenAI(temperature=1.0))
task_specifier_msg = task_specifier_template.format_messages(assistant_role_name=assistant_role_name,
                                                             user_role_name=user_role_name,
                                                             task=task, word_limit=word_limit)[0]
specified_task_msg = task_specify_agent.step(task_specifier_msg)
print(f"Specified task: {specified_task_msg.content}")
specified_task = specified_task_msg.content

assistant_inception_prompt = (
"""Your name is {assistant_role_name}. You are a professional software developer with special knowledge about python.
Keep you messages short, focus on the code.

Your task is to {task} with {user_role_name}.

Converse in this structure:

Response: <YOUR_RESPONSE>
Reflection: <YOUR_REFLECTION>
Code: <YOUR_CODE>
Critique: <YOUR_CRITIQUE>

When and only when the task is completed, please reply with <CAMEL_TASK_DONE>.
"""
)

user_inception_prompt = (
"""Your name is {user_role_name}. You are a professional software developer with special knowledge about python.
Keep you messages short, focus on the code.

Your task is to {task} with {assistant_role_name}.

Converse in this structure:

Response: <YOUR_RESPONSE>
Reflection: <YOUR_REFLECTION>
Code: <YOUR_CODE>
Critique: <YOUR_CRITIQUE>

When and only when the task is completed, please reply with <CAMEL_TASK_DONE>.
"""
)

def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
    
    assistant_sys_template = SystemMessagePromptTemplate.from_template(template=assistant_inception_prompt)
    assistant_sys_msg = assistant_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task)[0]
    
    user_sys_template = SystemMessagePromptTemplate.from_template(template=user_inception_prompt)
    user_sys_msg = user_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task)[0]
    
    return assistant_sys_msg, user_sys_msg

assistant_sys_msg, user_sys_msg = get_sys_msgs(assistant_role_name, user_role_name, specified_task)
assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(temperature=0.2))
user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(temperature=0.2))

# Reset agents
assistant_agent.reset()
user_agent.reset()

# Initialize chats 
assistant_msg = HumanMessage(content=f"{user_sys_msg.content}")

user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
user_msg = assistant_agent.step(user_msg)

print(f"Original task prompt:\n{task}\n")
print(f"Specified task prompt:\n{specified_task}\n")

chat_turn_limit, n = 10, 0
while n < chat_turn_limit:
    n += 1
    user_ai_msg = user_agent.step(assistant_msg)
    user_msg = HumanMessage(content=user_ai_msg.content)
    print(f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n")
    
    assistant_ai_msg = assistant_agent.step(user_msg)
    assistant_msg = HumanMessage(content=assistant_ai_msg.content)
    print(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")
    if "<CAMEL_TASK_DONE>" in user_msg.content:
        break