# LLMRoboFund Empowered with Retrieval-Augmented Generation 

Conducting investment research requires obtaining access to various platforms to become well-informed about investment strategies, financial risks involved, or even financial data about a fund or ETFs. Traditional chatbots, also called 'Robo-Funds,' offer mere hard-coded pre-defined choices based on your answers, being unable to generate or answer questions with real-time information or data. LLM-backed chatbots can easily eliminate the need for standalone research with dedicated hours.

LLMRoboFund presents a novel approach for chatting with an LLM to get informed about funds/ETFs and obtain financial data with up-to-date documents. To enable the LLM with real-time information, Retrieval-Augmented Generation (Lewis et al., 2020) method is deployed to update the knowledge base using dense vector and SQL databases.

The data used to update the knowledge base of the LLM include [Turkey Electronic Fund Trading Platform](https://www.tefas.gov.tr/) and [Public Disclosure Platform](https://www.kap.org.tr/tr/YatirimFonlari/BYF). TEFT platform provide diverse financial information, such as management fee, outstanding number of shares, initial and current price, return data up-to 5 year, percentage distribution of the invested instruments, and so on. PDP share documents related to the funds/ETFs available at TEFT platform, such as investor information documents, which are the foundation of LLMRoboFund.  

# Requirements

```Python
 streamlit==1.28.0
 tefas-crawler==0.3.4
 openai==0.28.1
 streamlit-dynamic-filters==0.1.3
 faiss-cpu==1.7.4
 streamlit==1.28.0
 cohere==4.36
 langchain==0.0.348
 pypdf==3.17.1
 chromadb==0.4.18
 pinecone-client==2.2.4
 psutil==5.9.6
 gputil==1.4.0
 tiktoken==0.5.2
 selenium==4.16.0
 chardet==5.2.0
 streamlit_option_menu==0.3.6
 streamlit_chat==0.1.1
 plotly==5.18.0
 tiktoken==0.5.2
```


```Python
   $Streamlit run Application/app.py
```


## Pinecone Schema
![image](https://github.com/dfavenfre/LLMRoboFund/assets/118773869/1f09a9dc-a9bc-4f05-a334-c4be57efbbbe)

 
## LangChain Schema
![image](https://github.com/dfavenfre/LLMRoboFund/assets/118773869/de739516-5ea9-48d5-911e-35d8dd9eb6cc)

# LLMRoboFund
## Investment Chatbot
[streamlit-app-2023-12-08-16-12-37.webm](https://github.com/dfavenfre/LLMRoboFund/assets/118773869/d7439c19-b018-4d8f-a8d2-1a73502efda2)

## Interactive Dashboard
 [streamlit-app-2023-12-04-12-12-38.webm](https://github.com/dfavenfre/LLMRoboFund/assets/118773869/0270edb3-9b4c-4347-a522-7e85bfe899a2)
