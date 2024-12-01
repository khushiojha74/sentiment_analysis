from fastapi import FastAPI
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pydantic import BaseModel

app = FastAPI()

class CommentInput(BaseModel):
    comments: list[str]

@app.post("/analyse-comments/")
async def analyse_comments(input: CommentInput):
    pos, neg, neutral = 0, 0, 0
    results = []

    for comment in input.comments:
        analyzer = SentimentIntensityAnalyzer()
        sentiment_dict = analyzer.polarity_scores(comment)
        sentiment_score = sentiment_dict['compound']

        if sentiment_score > 0:
            pos += sentiment_score
            sentiment_type = "Positive"
        elif sentiment_score < 0:
            neg += sentiment_score
            sentiment_type = "Negative"
        else:
            neutral += sentiment_score
            sentiment_type = "Neutral"

        results.append({
            "comment": comment,
            "sentiment_score": sentiment_score,
            "sentiment_type": sentiment_type,
        })

    overall_quality = max(pos, abs(neg), neutral)
    return {"results": results, "overall_quality": overall_quality}
