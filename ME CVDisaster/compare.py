import argparse
from geolocalization.comparison import compare_results


def main():
    parser = argparse.ArgumentParser(description="比较 seed=42 改进模型与作业 3979539 的已有基线")
    parser.add_argument("--run", required=True, help="改进模型的完整训练输出目录")
    parser.add_argument("--baseline-queries", required=True, help="已有基线的完整逐查询记录 CSV")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compare_results(args.run, args.baseline_queries, args.output)


if __name__ == "__main__":
    main()
