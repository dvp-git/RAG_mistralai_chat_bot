Rag-chat: A cli based pdf QnA assistant using mistral-ai-v2.0 , Faiss and all-mpnet-base-v2 sentence transformer

Note: The application requires a very high compute CPU/GPU even though it is a quantized version. I believe there are certain optimizations that can be performed, but the RAM memory requirement for me was about 16GB+. I used PaperSpace gradient notebooks while developing the application since there are free shared GPU machines available.

You can signup using your email and start building on free GPU's too!
https://www.paperspace.com/ 

Install and Usage:
------------------
1. Clone the directory to your local environment

```
git clone https://github.com/dvp-git/RAG_mistralai_chat_bot.git
```

2. Change directory to *RAG_mistralai_chat_bot* and install the necessary libraries using requirements.txt. Note: Use the latest version of python, preferably >=3.12

 ```
 pip install -r requirements.txt
 ```

3. Upload the PDF documents you'd like to receive answers from in the project directory.

Eg: I have uploaded the famous Attention is all you need, FAISS, Gemini and the Stanford NLP book pdf's. Make sure the books you upload are publicly available and not confidential documents.

![image](https://github.com/dvp-git/RAG_mistralai_chat_bot/assets/43114889/248c808b-70c9-494b-96ff-b7f8685f44e1)
 
4. Once done, run the application in terminal using

 ```
 python app.py
 ```
 ![image](https://github.com/dvp-git/RAG_mistralai_chat_bot/assets/43114889/694f738b-470c-4f6b-9883-3f5457b3d76a)

5. Ask questions about the pdf and you should receive an answer.
 
Context based
![image](https://github.com/dvp-git/RAG_mistralai_chat_bot/assets/43114889/ed99d519-42b0-4f80-a669-7ae31d61e59f) -

Contextless question:
![image](https://github.com/dvp-git/RAG_mistralai_chat_bot/assets/43114889/b43782d5-3dcb-4ddb-953e-65d612c185fd)  

It is likely you will get answers to only context based questions, i.e. the answers are somewhere in the pdf. However you can always increase the temperature of the llm and have more randomness in the generated output, so go forth and experiment!


6. To exit , type : `quit` or `exit` and hit Enter.
 
The application is experimental since I wanted to build a RAG application from scratch without using LangChain or the amazing libraries available. 
I think generative AI is great once you start learning the intricate details of what actually goes on under the hood.


Resources :
------------
Krish Naik - Videos on Building RAG, generative AI
DeepLearning.ai : Awesome courses to get you started on generative AI and deep learning.
How to build a RAG from scratch - 



