# LLM-s Rockybot 
Rockybot is a chatbot where it uses OpenAI pai to retrieve answers to the question from the given URL links.

# Steps followed
* URL are loaded using UnstructuredURLLoader class 
*  the loaded data is then split into chunks using RecursiveSPlitter class
* Then the chunks re converetd into embeddings using OpenAI embeddings
* Now to retrieve the query faster we used FASSAI as a vector index which gives the result faster





https://github.com/chinmay002/LLM-s/assets/60249099/8757845d-f222-4f69-88d7-f8b16b26634b





