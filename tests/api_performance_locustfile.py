from locust import HttpUser, between, task


class QuickstartUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def get_root(self):
        self.client.get("/")

    @task(4)
    def get_predict_labels_one_tweet(self):
        self.client.post("/predict_labels_one_tweet?tweet=lovetweet")

    @task(2)
    def predict_labels_tweets_file(self):
        # Create a file with tweets
        tweets = "I love you\nI hate you"
        with open("locust_test_file.txt", "w") as f:
            f.write(tweets)

        # Send the file
        with open("locust_test_file.txt", "rb") as f:
            self.client.post("/predict_labels_tweets_file", files={"file": f})

    def on_stop(self):
        """
        Clean up the file created during the test
        """
        import os

        # Check if the file exists before attempting to remove it
        if os.path.exists("locust_test_file.txt"):
            os.remove("locust_test_file.txt")
