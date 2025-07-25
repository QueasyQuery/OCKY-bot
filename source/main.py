import json
from sentence_transformers import SentenceTransformer
import asyncio

from bot_logic import DiscordHandler
from response_sys import ResponseSystem
from choice_sys import ChoiceSystem
from training import TrainingManager
from data import DataManager

class OCKYBot:
    def __init__(self, config_file="config.json"):
        # load config, responses and transformer
        self.config = self.load_json(config_file)
        self.responses = self.load_json(self.config.get('response_file', 'responses.json'))
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # initialize systems
        self.data_manager = DataManager(self.config)
        self.training_manager = TrainingManager(self.config, self.data_manager)
        self.response_system = ResponseSystem(self.config, self.data_manager, self.sentence_model)
        self.choice_system = ChoiceSystem(self.config, self.data_manager, self.sentence_model, self.responses)
        self.discord_handler = DiscordHandler(self.config, self.data_manager, self)

        # misc
        self.bot = self.discord_handler.bot
        self.channel_id = self.config['channel']

    def load_json(self, filename):
        '''load dict from JSON file'''
        with open(filename, 'r',encoding="utf-8") as f:
            return json.load(f)
    
    def run(self, token):
        '''Start the bot'''
        self.bot.run(token)

    async def shutdown(self):
        print("shutting down...")
        if self.config['training'] == 1:
            channel = self.bot.get_channel(self.channel_id)
            await channel.edit(topic='**STATUS: OFFLINE**')
        await self.bot.close()

if __name__ == "__main__":
    # read token
    with open('TOKEN.txt', 'r') as f:
        token = f.readline().strip()

    # run
    print("initializing...")
    bot = OCKYBot()
    try:
        bot.run(token)
    except Exception as e:
        print(f"Error: {e}")
        asyncio.run(bot.shutdown())