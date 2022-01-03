from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--maxlen_train', type=int, default=30, help='Maximum number of tokens in the input sequence during training')
parser.add_argument('--maxlen_val', type=int, default=50, help='Maximum number of tokens in the input sequence during evaluate')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate for Adam.')
parser.add_argument('--num_eps', type=int, default=2, help='Number of training epochs.')
parser.add_argument('--num_threads', type=int, default=0, help='Number of threads for collecting the datasets.')
parser.add_argument('--save_dir', type=str, default='my_model', help='Where to save the trained model, if relevant.')
parser.add_argument(
	'--text_model_name_or_path',
	type=str,
	default=None,
	help='''Name of or path to the pretrained/trained model. 
			For training choose between bert-base-uncased, albert-base-v2, distilbert-base-uncased etc. 
			For evaluating/analyzing/server choose between barissayil/bert-sentiment-analysis-sst and paths to the models you have trained previously.'''
)
parser.add_argument('--image_model_path', type=str, default=None)
parser.add_argument('--image_features_extractor', type=str, default="vit")
parser.add_argument('--classification_model_path', type=str, default=None)
parser.add_argument('--train_file', type=str)
parser.add_argument('--val_file', type=str)
parser.add_argument('--test_file', type=str)
parser.add_argument('--mode', type=str)
parser.add_argument('--text_column_name', type=str)
parser.add_argument('--label_column_name', type=str)
parser.add_argument('--image_column_name', type=str, default='image_id')
parser.add_argument('--train_image_folder', type=str, default='./train')
parser.add_argument('--val_image_folder', type=str, default='./validation')
parser.add_argument('--test_image_folder', type=str, default='./test')

args = parser.parse_args()