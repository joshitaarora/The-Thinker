## Refer to the sample notebook for the implementation details

from dotenv import load_dotenv
import os

from pymilvus import MilvusClient

load_dotenv()


client = MilvusClient(uri=os.environ.get("MIILVUS_SERVER"),  token=os.environ.get("MIILVUS_TOKEN"))


def get_client():
    return client
