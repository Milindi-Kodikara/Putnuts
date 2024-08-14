# @author Milindi Kodikara, RMIT University, 2024

import sys
import praw
import os
from dotenv import load_dotenv

load_dotenv()


def client():
    """
        Setup Reedit API authentication.
        @returns: praw Reddit object
    """

    try:
        clientId = os.environ["CLIENT-ID"]
        clientSecret = os.environ["CLIENT-SECRET"]
        password = os.environ["PASSWORD"]
        userName = os.environ["USERNAME"]
        userAgents = os.environ["USER-AGENTS"]

        reddit_client = praw.Reddit(client_id=clientId,
                             client_secret=clientSecret,
                             password=password,
                             username=userName,
                             user_agent=userAgents)
    except KeyError:
        sys.stderr.write("Key or secret token are invalid.\n")
        sys.exit(1)

    return reddit_client
