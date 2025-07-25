# O.C.K.Y. bot
**Overly Complicated Keras Yapper***

OCKY is meant to be a entertaining discord server bot running fully locally using python. While it only has a set list of responses, it is able to react to any message, due to two systems: one that predicts whether or not to react to a message, and one that links the tokens of the sent message into a choice of message to send. 

The main feature is that it learns through crowdwork: people invoke OCKY to respond by reacting to a message it should respond to. And vote whether or not a returned message is good by reacting accordingly to a message. The bot then incorporates this feedback into it's neural networks. In simple terms, this bot learns when to respond, and what to respond, in realtime on the server. At the start, the bot is very, very dumb.

### Deploying OCKY-bot
While I don't recommend doing this as it's kind of really made for me, and the system is ass, you can deploy an instance of OCKY-bot by:
1. Cloning the repository
```
git clone https://github.com/QueasyQuery/OCKY-bot
```
2. Running run.bat
3. Change responses.json to whatever you want and restart the system.


##
*Yes I know this doesn't use KERAS. It did at some point, name stuck. Oops.