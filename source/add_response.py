import discord
from discord.ext import commands
from discord import app_commands
import hashlib
import importlib

class ResponseAdder(commands.Cog):
    def __init__(self, bot, ocky_bot, data):
        self.bot = bot # (discord bot instance)
        self.ocky_bot = ocky_bot # (OCKY_bot instance) 
        self.data = data
        self.config = ocky_bot.config
        self.responses_file = ocky_bot.config.get('response_file', 'responses.json')
        self.setup_commands()

    def setup_commands(self):
        @self.bot.tree.command(name="ocky_response", description="Add new response to OCKY-bot")
        @app_commands.describe(
            category="Category of response",
            response="Response text",
            example_input="Example of input that would trigger this response"
        )
        async def ocky_response(interaction: discord.Interaction, category: str, response: str, example_input: str):
            '''Add a new response to the bot'''
            await interaction.response.defer(ephemeral=True) # make response private

            # load current responses
            data = self.ocky_bot.responses

            # initialize category if it doesn't exist
            if category not in data:
                await interaction.followup.send(f"! Category '{category}' doesn't exist. Use `/ocky_categories` to list them.")
                return

            # check if response already exists
            if response in data[category]['responses']:
                await interaction.followup.send(f"! Response already exists in category '{category}'",ephemeral=True)
                return
            
            # add the response to the responses object and save back to file
            data[category]['responses'].append(response)
            self.data.save_responses(data)

            # add embedding to this response manually
            response_hash = hashlib.md5(response.encode('utf-8')).hexdigest()[:10]
            response_id = (category, response_hash)
            
            # create embedding and save
            embeddings = self.ocky_bot.choice_system.response_embeddings
            new_embedding = self.ocky_bot.sentence_model.encode(example_input)
            embeddings[response_id] = new_embedding
            self.ocky_bot.choice_system.response_dict[response_id] = response
            succes = self.data.save_model(self.ocky_bot.choice_system.response_embeddings, 'response_embeddings.pkl')
            
            message = f"! Added response to category '{category}'" if succes else "! Failed to save responses file"
            await interaction.followup.send(message, ephemeral=True)
            print(f'{interaction.user.nick} added "{response}" to {category} with embedding "{example_input}"')

        @self.bot.tree.command(name="ocky_categories", description="List all categories for OCKY-bot responses.")
        async def ocky_categories(interaction: discord.Interaction):
            '''List all available categories'''
            await interaction.response.defer(ephemeral=True)

            data = self.ocky_bot.responses
            embed = discord.Embed(title="Response Categories", color=0x00ff00)

            for category, category_data in data.items():
                count = len(category_data.get('responses', []))
                example = category_data.get('example_input', 'No example')
                embed.add_field(
                    name=f"{category} ({count} responses)", 
                    value=f"Example: {example[:50]}{'...' if len(example) > 50 else ''}", 
                    inline=False
                )
            
            await interaction.followup.send(embed=embed, ephemeral=True)

        @self.bot.tree.command(name="ocky_reloader", description="Reload ocky modules")
        @app_commands.describe(module="what module to reload")
        async def ocky_reloader(interaction: discord.Interaction, module:str="training"):
            await interaction.response.defer(ephemeral=True)
            self.ocky_bot.reload(module)
            await interaction.followup.send(f"reloaded {module}!", ephemeral=True)
                    

