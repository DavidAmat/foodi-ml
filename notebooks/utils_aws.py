import boto3
import botocore
import copy
import os

class AWSConnector:

    def __init__(self, bucket):
        session = boto3.Session()
        self.s3_client = session.client("s3")
        self.s3_resource = session.resource('s3')
        self.bucket = bucket

class AWSBasics:

    def __init__(self, bucket):
        self.aws_con = AWSConnector(bucket)
        self.bucket = bucket


    def download_obj(self, s3_key, destination):
        try:
            self.aws_con.s3_client.download_file(
                self.bucket,
                s3_key,
                destination
            )
        except botocore.exceptions.ClientError as error:
            print(f"Key {s3_key} not found in S3")
            return False

        return True

    def download_subfolders(self, s3_key):
        list_folders = self.aws_con.s3_client.list_objects_v2(
            Bucket = self.bucket,
            Prefix = s3_key,
            Delimiter='/'
        )
        return list_folders

class AWSTools:
    def __init__(self, aws_con):
        self.aws_con = aws_con
        self.aws_basics = AWSBasics(self.aws_con.bucket)

    def create_list_cities(self, s3_key):
        # List subfolders of s3_key
        list_folders = self.aws_basics.download_subfolders(s3_key)

        # Parse cities
        l_cities = []
        if list_folders.get("CommonPrefixes") is not None:
            l_key_cities = list_folders.get("CommonPrefixes")
            for key in l_key_cities:
                if key.get("Prefix") is not None:
                    l_cities.append(key["Prefix"].split("/")[:-1].pop())
        return l_cities

    def downloading_city_csv(self, l_cities, s3_key_prefix, csv_name,
                             local_folder,
                             verbose=False):
        """
        Downloads from S3 the csv of each of the cities in l_cities

        Parameters
        ----------
        l_cities : list
            List of cities to download the training_data.csv
            Ex: ["CUG", "BCN"]
        s3_key_prefix : str
            Prefix of the path to the images
            Ex: artifacts/002/
        csv_name: file name of the S3 csv for a given city
            Ex: by default training_data.csv
        local_folder: local folder in which .csv will be saved
            Ex: /home/ec2-user/SageMaker/tmp/samples
        verbose
        Returns
        -------
        out_cities : list
            List of cities, removing the ones which we could not download
            from S3 the .csv file
        """

        assert len(l_cities) > 0, "Length of l_cities should be > 0"

        # Ensure folder where cities will be downloaded exist
        os.makedirs(local_folder, exist_ok=True)

        # Make a copy to modify cities if cannot be downloaded
        out_cities = copy.copy(l_cities)

        # For each city download the csv to the destination (full path)
        for city in l_cities:

            # Get the prefix of the S3 csv name
            local_file_name = f"{city}.csv"

            # Get full local path
            local_file_path = os.path.join(local_folder, local_file_name)

            # S3 key of the .csv
            s3k_file_path = os.path.join(s3_key_prefix, city, csv_name)

            # Download csv
            success = self.aws_basics.download_obj(
                s3_key=s3k_file_path,
                destination=local_file_path
            )
            if not success:
                if verbose:
                    print(f"Removing from l_cities city {city}")
                out_cities.remove(city)
            else:
                if verbose:
                    print(f"City {city} correctly downloaded to {local_file_path}")

        return out_cities


class ImageDownloaderParallelS3:

    def __init__(self, base_path,):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    @staticmethod
    def initialize(bucket):
        global con_parallel
        con_parallel = AWSConnector(bucket)

    def create_jobs(self, samples):
        jobs = []

        # Keep appending image names to the jobs list
        for img_url in samples["image_names_s3"].to_list():
            img_s3_key = img_url.split("/")[-1]
            full_name = os.path.join(self.base_path, img_s3_key)
            jobs.append((img_url, full_name))

        # Modify img_path to the local instance path
        samples["img_path"] = samples["image_names_s3"].apply(
            lambda x: os.path.join(self.base_path, x.split("/")[-1])
        )
        return jobs, samples

    @staticmethod
    def download_images(job):
        img_url, full_name = job
        if not os.path.isfile(full_name):
            try:
                with open(full_name, "wb") as f_handle:
                    con_parallel.s3_client.download_fileobj(
                        con_parallel.bucket,
                        img_url,
                        f_handle
                    )
            except:
                return False
        return True


class ImageDownloaderS3:

    def __init__(self, bucket, base_path):
        self.aws_con = AWSConnector(bucket)
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def download_imgs_in_disk(self, samples):
        """
        Ingests a dataframe containing URLs of images and saves a resized version. If there is any problem while
        downloading an image, it removes the row from the dataframe.
        Parameters
        ----------
        samples : pd.DataFrame
            DataFrame containing at least "image_service_id" containing the URLs of the images.
            Columns: ["image_service_id", ...]
        Returns
        -------
        samples : pd.DataFrame
            Modified DataFrame with images that have been SUCCESSFULLY downloaded into disk.
            One column containing the path of the images in disk is added: img_path
            Columns: ["image_service_id", ...,"img_path"]
        """
        for img_url in samples["image_names_s3"].to_list():
            img_s3_key = img_url.split("/")[-1]
            full_name = os.path.join(self.base_path, img_s3_key)
            if not os.path.isfile(full_name):
                try:
                    with open(full_name, "wb") as f:
                        self.load_img_from_s3(img_url, f)
                except Exception:
                    # Remove that specific image in case it fails
                    # product_id = "/".join(img_url.split("/")[-2:])  # BUG
                    # print("Could not save image! ", product_id, img_url)
                    # TODO: ask Ponç if this should be image_names_s3
                    samples = samples[~(samples["image_names_s3"] == img_url)].copy()

        # We know that the image_names_s3 contains the names structured as:
        # artifacts/version/city_code/images/0000016_0000026_350899836.png
        samples["img_path"] = samples["image_names_s3"].apply(
            lambda x: os.path.join(self.base_path, x.split("/")[-1])
        )
        return samples

    def load_img_from_s3(self, s3_key, destination):
        self.aws_con.s3_client.download_fileobj(
            self.aws_con.bucket, s3_key, destination
        )

    def remove_images_from_disk(self, samples):
        """
        Deletes images in samples dataframe.
        Parameters
        ----------
        samples : pd.DataFrame
            DataFrame containing at least the path of the images to be deleted.
            Columns: ["img_path", ...]
        """
        for img_path in samples["img_path"].to_list():
            os.remove(img_path)