import boto3


s3 = boto3.client(
    service_name="s3",
    region_name="us-east-1",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    endpoint_url="http://localhost:4566",
)


def create_bucket():
    try:
        s3.create_bucket(Bucket="ship-model-artifacts")
        print(" Bucket created.")
    except Exception as e:
        print(" Bucket may already exist:", e)


def upload_model():
    s3.upload_file("models/model.pkl", "ship-model-artifacts", "model.pkl")
    print(" Model uploaded.")


def download_model():
    s3.download_file("ship-model-artifacts", "model.pkl", "models/downloaded_model.pkl")
    print(" Model downloaded.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["create", "upload", "download"])
    args = parser.parse_args()

    if args.action == "create":
        create_bucket()
    elif args.action == "upload":
        upload_model()
    elif args.action == "download":
        download_model()
