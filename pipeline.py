from google.cloud import storage
from google.cloud import pubsub_v1

import torch
import os
import json

PROJECT_ID=os.environ["PROJECT_ID"]
BUCKET_NAME="csci-ga-2752-dl-jepa"
DL_EVAL_TOPIC="dl-eval-topic"
DL_EVAL_SUBSCRIPTION="dl-eval-topic-sub"
DL_LOCAL_PATH="/tmp/dl-eval/checkpoints"

class CheckpointEvalPipeline:
    def __init__(self):
        self.storage_client: storage.Client = storage.Client(project=PROJECT_ID)
        self.bucket: storage.Bucket = self.storage_client.bucket(BUCKET_NAME)

        self.publish_topic = 'projects/{project_id}/topics/{topic}'.format(
            project_id=PROJECT_ID,
            topic=DL_EVAL_TOPIC, 
        )
        self.subscribe_topic = 'projects/{project_id}/subscriptions/{sub}'.format(
            project_id=PROJECT_ID,
            sub=DL_EVAL_SUBSCRIPTION, 
        )
        self.publisher_client = pubsub_v1.PublisherClient()
        self.consumer_client = pubsub_v1.SubscriberClient()

    def health_check(self):
        if self.publisher_client.get_topic(topic=self.publish_topic) is None:
            raise Exception(f"PubSub topic {self.publish_topic} does not exist.")
        
        if self.consumer_client.get_subscription(subscription=self.subscribe_topic) is None:
            raise Exception(f"PubSub subscription {self.subscribe_topic} does not exist.")

        if self.storage_client.get_bucket(BUCKET_NAME) is None:
            raise Exception(f"Storage bucket {BUCKET_NAME} does not exist.")


    # Subscribe to the checkpoint topic. `process_download_checkpoint` is a callback function
    # that will be called after the checkpoint is downloaded to the local directory.
    def subscribe_checkpoint(self, process_download_checkpoint):
        self.consumer_client.subscribe(DL_EVAL_SUBSCRIPTION, 
                                       self.__process__checkpoint_internal__(process_download_checkpoint))
        pass

    def __process__checkpoint_internal__(self, process_download_checkpoint):
        def callback(message):
            try:
                payload = json.loads(message.data)
                checkpoint_url = payload.get("checkpoint_url")
                if checkpoint_url:
                    # Download the blob to a local file
                    blob = self.bucket.blob(checkpoint_url)
                    local_file_path = os.path.join(DL_LOCAL_PATH, os.path.basename(checkpoint_url))
                    blob.download_to_filename(local_file_path)

                    # Process the downloaded file
                    print(f"Downloaded checkpoint to {local_file_path}")
                    process_download_checkpoint(local_file_path)
                else:
                   raise ValueError("No checkpoint URL found in the message.") 
            except Exception as e:
                print(f"Error processing message: {e}")
            finally:
                message.ack()

        return callback


    # Creates a checkpoint URL with a blob_save_callback to write file to the blob
    # When the blob is written, it will publish a message to the PubSub topic.
    def produce_checkpoint(self, epoch, blob_save_callback):
        file_path = f'checkpoints/model_epoch_{epoch}.pth' 
        blob = self.bucket.blob(file_path)

        try:
            blob_save_callback(blob)
            print(f"Blob saved successfully to {blob.public_url}")
        except Exception as e:
            print(f"Error saving blob: {e}")
            raise e

        # pubsub publish if the blob write is correct
        print("Publish checkpoint to PubSub")
        try:
            publish_payload = {"checkpoint_url": blob.public_url}
            self.publisher_client.publish(DL_EVAL_TOPIC, 
                                          data=json.dumps(publish_payload).encode('utf-8')) 
            print("Published checkpoint to PubSub")
        except Exception as e:
            print(f"Error publishing to PubSub: {e}")
            raise e

        return blob 

# Some tests

if __name__ == "__main__":
    pipeline = CheckpointEvalPipeline()
    pipeline.health_check()
    print("Health check passed.")
