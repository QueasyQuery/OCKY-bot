import time
import discord
from discord import app_commands
from discord.ext import commands, tasks
from add_response import ResponseAdder
import asyncio

class DiscordHandler():
    ''' handles all logic pertaining to discord communications (receiving/sending information to the server).
    '''
    def __init__(self, config, data_manager, ocky_bot):
        # vars
        self.config = config
        self.is_training = config.get('training', 0) == 1
        self.channel_id = self.config.get('channel')
        self.data_manager = data_manager
        self.ocky_bot = ocky_bot  # Reference to main bot instance

        # bot setup
        intents = discord.Intents.all()
        self.bot = commands.Bot(command_prefix='!', intents=intents)
        self._setup_events()
    
    async def send_response(self, message, response):
        print(f"responding with {response['text']}")
        sent_message = await message.channel.send(response['text'])

        # react self
        for emoji in ['🟩','👍','👎','🟥']: await sent_message.add_reaction(emoji)

        # track data for this response
        self.data_manager.prev_response[message.channel.id] = time.time()
        self.data_manager.record_bot_response(
            message, sent_message, response['id'], self.ocky_bot.sentence_model
        )

    def _setup_events(self):
        '''define bot events'''
        @self.bot.event
        async def on_ready():
            print(f'OCKY system active as {self.bot.user}')
            # init response adder
            await self.bot.add_cog(ResponseAdder(self.bot,self.ocky_bot,self.data_manager))
            synced = await self.bot.tree.sync()
            print(f"Synced {len(synced)} slash commands")

            # training loop & finish
            self.training_loop.start()
            if self.channel_id: 
                self.channel = self.bot.get_channel(self.channel_id)
                if self.channel:
                    await self.channel.edit(topic='**STATUS: ONLINE**')

        @self.bot.event
        async def on_message(message: discord.Message):
            try:
                if message.author.bot: return
                
                # process commands first
                await self._check_commands(message)
                
                print(f'message got: {message.content}')
                
                # track message activity
                self.data_manager.track_channel_activity(message)

                # ask response system: should we respond?
                response_chance = self.ocky_bot.response_system.should_respond(message, self.bot.user)

                # check if we should only respond in specific channel
                if self.channel_id and (message.channel.id != self.channel_id): return
                
                # decide to respond based on probability
                if response_chance > 0.5:  # 50% threshold
                    # ask choice system: what to respond with?
                    response = self.ocky_bot.choice_system.get_response(message)
                    if response: await self.send_response(message, response)
            # always unload to end with
            finally: self.ocky_bot.choice_system.sentence_model.unload()

        @self.bot.event
        async def on_reaction_add(reaction, user):
            if user.bot: return
            # if emote is "forceful reaction emote" then respond
            if reaction.emoji == self.config['forceful_react_emote']:
                print("Respond Emote Received.")
                message = reaction.message
                response = self.ocky_bot.choice_system.get_response(message)
                if response: 
                    await self.send_response(message, response)
                self.ocky_bot.choice_system.sentence_model.unload()

            # process feedback
            if reaction.emoji in ["👍","👎","🟩","🟥"]:
                value = await self.data_manager.process_feedback(reaction, user, is_add=True, bot_user=self.bot.user)
                print(f"Feedbacd incorporated ({value})")
        
        @self.bot.event
        async def on_reaction_remove(reaction, user):
            if user.bot: return
            await self.data_manager.process_feedback(reaction, user, is_add=False, bot_user=self.bot.user)

        @tasks.loop(hours=1)  # try to train every hour by default
        async def training_loop():
            try:
                await self.ocky_bot.training_manager.training_loop(self.ocky_bot.response_system,self.ocky_bot.choice_system)
            except Exception as e:
                print(f"Training error: {e}")

        # make it callable too
        self.training_loop = training_loop

    async def _check_commands(self, message):
        content = message.content.lower()

        if "retrain" in content and "models" in content:
            print("Manual retraining...")
            await self.ocky_bot.training_manager.training_loop(
                self.ocky_bot.response_system,
                self.ocky_bot.choice_system,
                message
            )
        elif "shutdown" in content and any(phrase in content for phrase in ['pls','thx','aub','thanks','please','thank']):
            await self.ocky_bot.shutdown()