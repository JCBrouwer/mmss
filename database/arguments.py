import argparse
import os
from features.registry import REGISTRY


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir", type=str, default="cache/db", help="Directory to save database indices")
    parser.add_argument(
        "--features", nargs="+", default=["clip"], choices=["all", *REGISTRY.keys()], help="Features to process"
    )
    return parser


def parse_insert_args():
    parser = init_parser()
    parser.add_argument("img_dir", type=str, help="Directory with images to insert to the database")
    parser.add_argument("--batch_size", type=int, default=8, help="Size of batches to process features in")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of feature processing threads to use")
    return parser.parse_args()


def parse_search_args():
    parser = init_parser()
    parser.add_argument("--num_results", type=int, default=8, help="Number of results to return")
    parser.add_argument("--filenames_only", action="store_true", help="Only output filenames of results, do not show images")
    return parser.parse_args()
