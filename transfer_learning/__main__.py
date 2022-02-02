import sys
from .train import main
from .parse_args import parse_args


MODEL_DIR = "./cnn_model.h5"

if __name__ == "__main__":
    print("weird. goes into __main__.py")
    args = parse_args(sys.argv[1:])
    MODEL_DIR = args["model"] or MODEL_DIR
    main(args["images"], args["labels"], MODEL_DIR, args["weights"], file_filter=args["filter"])
