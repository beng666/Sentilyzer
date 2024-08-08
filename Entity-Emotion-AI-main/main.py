import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from utils import entity_context_analysis, predict_sentiment, model

app = FastAPI()


class Item(BaseModel):
    text: str = Field(...,
                      example="""Fiber 100mb SuperOnline kullanıcısıyım yaklaşık 2 haftadır @Twitch @Kick_Turkey gibi canlı yayın platformlarında 360p yayın izlerken donmalar yaşıyoruz. Başka hiç bir operatörler bu sorunu yaşamazken ben parasını verip alamadığım hizmeti neden ödeyeyim ? @Turkcell""")


@app.post("/predict/", response_model=dict)
async def predict(item: Item):
    text = item.text
    print(f"Received text: {text}")

    try:
        entity_context_map = entity_context_analysis(text)
        print(f"Entity and context map: {entity_context_map}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Entity ve bağlam analizi sırasında bir hata oluştu: {e}")

    results = {
        "entity_list": [],
        "results": []
    }

    seen_entities = set()  # Aynı entity ve sentiment kombinasyonunu bir kez eklemek için set

    for entity_text, data in entity_context_map.items():
        contexts = data['contexts']
        print(f"Processing entity: {entity_text}, contexts: {contexts}")

        for context in contexts:
            try:
                predicted_sentiment = predict_sentiment(context, model)
                print(f"Predicted sentiment for context '{context}': {predicted_sentiment}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Sentiment tahmini sırasında bir hata oluştu: {e}")

            # Aynı entity ve sentiment kombinasyonu daha önce görülmemişse, results listesine ekle
            entity_sentiment_key = (entity_text, predicted_sentiment)
            if entity_sentiment_key not in seen_entities:
                seen_entities.add(entity_sentiment_key)
                if entity_text not in results["entity_list"]:
                    results["entity_list"].append(entity_text)
                results["results"].append({
                    "entity": entity_text,
                    "sentiment": predicted_sentiment
                })

    print(f"Final results: {results}")
    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
