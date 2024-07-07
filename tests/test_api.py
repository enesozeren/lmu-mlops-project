from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_read_root():
    """
    Test the root endpoint by getting the response.
    """
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the Hate Speech Detection model API!"}


def test_predict_labels_one_tweet():
    """
    Test the predict_labels_one_tweet endpoint by posting a tweet.
    """
    with TestClient(app) as client:
        response = client.post("/predict_labels_one_tweet?tweet=lovetweet")
        assert response.status_code == 200
        # Check if response is not-hate or hate
        assert response.json() in ["not-hate", "hate"]


def test_predict_labels_tweets_file():
    """
    Test the predict_labels_tweets_file endpoint by posting a file with tweets.
    """
    with TestClient(app) as client:
        # Create a file with tweets
        tweets = "I love you\nI hate you"
        with open("api_unittest_tweet_file.txt", "w") as f:
            f.write(tweets)

        # Post the file
        with open("api_unittest_tweet_file.txt", "rb") as f:
            response = client.post("/predict_labels_tweets_file", files={"file": f})

        assert response.status_code == 200
        # Check if response is not-hate or hate
        assert response.json() == ["not-hate", "hate"], "Response from the model is not as expected."

        # Remove the file
        import os

        if os.path.exists("api_unittest_tweet_file.txt"):
            os.remove("api_unittest_tweet_file.txt")
