import csv
import logging
from predictions_class import Predictions

class EvaluateModel:
    def __init__(self, corpus_manager, split_type=None, log_initialization_details=True):
        self.corpus_manager = corpus_manager
        self.corpus_name = corpus_manager.corpus_name
        self.config = corpus_manager.config
        self.model = corpus_manager.model
        self.corpus = corpus_manager.corpus
        self.train_set = corpus_manager.train_set
        self.test_set = corpus_manager.test_set
        self.all_words = corpus_manager.all_words
        self.split_type = split_type

        # Extract unique characters from the corpus
        unique_characters = corpus_manager.extract_unique_characters()
        self.unique_character_count = len(unique_characters)

        # Initialize prediction class
        self.q_range = range(self.config.q_range[0], self.config.q_range[1] + 1)
        self.predictor = Predictions(self.model, self.q_range, unique_characters)

        # Logging model initialization details
        if log_initialization_details:
            self.log_initialization_details()

        # Retrieve the appropriate prediction method
        self.prediction_method = self.get_prediction_method()

    def get_prediction_method(self):
        prediction_methods = {
            'context_sensitive': self.predictor.context_sensitive,
            'context_no_boundary': self.predictor.context_no_boundary,
            'base_prediction': self.predictor.base_prediction
        }
        return prediction_methods.get(self.config.prediction_method_name, self.predictor.context_sensitive)

    def log_initialization_details(self):
        logging.info(f'Language Model for {self.corpus_name} initialized with:')
        logging.info(f'Seed: {self.config.seed}')
        logging.info(f'Q-gram Range: {self.config.q_range}')
        logging.info(f'Train-Test Split Configuration: {self.config.split_config}')
        logging.info(f'Training Set Size: {len(self.train_set)}')
        logging.info(f'Testing Set Size: {len(self.test_set)}')
        logging.info(f'Vowel Replacement Ratio: {self.config.vowel_replacement_ratio}')
        logging.info(f'Consonant Replacement Ratio: {self.config.consonant_replacement_ratio}')
        logging.info(f'Unique Character Count: {self.unique_character_count}')
        logging.info(f'Minimum Word Length: {self.config.min_word_length}')
        logging.info(f'Number of Replacements per Word: {self.config.num_replacements}')

    def compute_metrics(self, predictions) -> dict:
        accuracy_counts = {1: 0, 2: 0, 3: 0}
        validity_counts = {1: 0, 2: 0, 3: 0}
        total_test_words = len(self.test_set)

        for modified_word, missing_letters, _, all_predictions, _ in predictions:
            if not all(isinstance(pred, tuple) and len(pred) == 2 for pred in all_predictions):
                logging.error(f'Invalid prediction format: {all_predictions}')
                continue

            correct_rank = next((rank for rank, (predicted_letter, _) in enumerate(all_predictions, start=1)
                                if predicted_letter in missing_letters), None)

            if correct_rank:
                for rank in range(correct_rank, 4):
                    accuracy_counts[rank] += 1

            valid_word_checked = [False] * 3

            for rank, (predicted_letter, _) in enumerate(all_predictions[:3], start=1):
                if not valid_word_checked[rank-1]:
                    reconstructed_word = modified_word.replace('_', predicted_letter, 1)
                    if reconstructed_word in self.all_words:
                        for i in range(rank, 4):
                            if not valid_word_checked[i-1]:
                                validity_counts[i] += 1
                                valid_word_checked[i-1] = True

        total_accuracy = {k: accuracy_counts[k] / total_test_words for k in accuracy_counts}
        total_validity = {k: validity_counts[k] / total_test_words for k in validity_counts}

        return {'accuracy': total_accuracy, 'validity': total_validity, 'total_words': total_test_words}

    def evaluate_character_predictions(self, prediction_method) -> tuple[dict, list]:
        predictions = []

        for modified_word, target_letters, original_word in self.test_set:
            try:
                all_predictions = prediction_method(modified_word)
                if not isinstance(all_predictions, list) or not all(isinstance(pred, tuple) and len(pred) == 2 for pred in all_predictions):
                    logging.error(f'Unexpected prediction format for {modified_word}: {all_predictions}')
                    continue

                correct_letter_rank = next((rank for rank, (retrieved_letter, _) in enumerate(all_predictions, start=1)
                                            if retrieved_letter in target_letters), None)

                predictions.append((modified_word, target_letters, original_word, all_predictions[:3], correct_letter_rank))
            except Exception as e:
                logging.error(f"Error predicting for {modified_word}: {e}", exc_info=True)

        evaluation_metrics = self.compute_metrics(predictions)
        return evaluation_metrics, predictions

    def save_summary_stats_txt(self, evaluation_metrics, predictions, prediction_method_name):
        output_file_path = self.config.text_dir / f'{self.corpus_name}_predictions.txt'
        try:
            with output_file_path.open('w', encoding='utf-8') as file:
                file.write(f'Prediction Method: {prediction_method_name}\n')
                file.write(f'Unique Character Count: {self.unique_character_count}\n\n')

                accuracy = evaluation_metrics['accuracy']
                validity = evaluation_metrics['validity']
                file.write(f'TOP1 ACCURACY: {accuracy[1]:.2%}\n')
                file.write(f'TOP2 ACCURACY: {accuracy[2]:.2%}\n')
                file.write(f'TOP3 ACCURACY: {accuracy[3]:.2%}\n')
                file.write(f'TOP1 VALIDITY: {validity[1]:.2%}\n')
                file.write(f'TOP2 VALIDITY: {validity[2]:.2%}\n')
                file.write(f'TOP3 VALIDITY: {validity[3]:.2%}\n\n')

                file.write(f'Train Size: {len(self.train_set)}, Test Size: {len(self.test_set)}\n')
                file.write(f'Vowel Ratio: {self.config.vowel_replacement_ratio}, '
                           f'Consonant Ratio: {self.config.consonant_replacement_ratio}\n\n')

                for mod_word, miss_letters, orig_word, top_preds, cor_letter_rank in predictions:
                    file.write(f'Test Word: {mod_word}, Correct Letters: {",".join(miss_letters)}\n')
                    file.write(f'Correct Letter Rank: {cor_letter_rank}\n')

                    for rank, (pred_letter, confidence) in enumerate(top_preds, start=1):
                        reconstructed_word = mod_word.replace('_', pred_letter)
                        is_valid_word = reconstructed_word in self.all_words

                        file.write(f"Rank {rank}: '{pred_letter}' (Confidence: {confidence:.8f}), "
                                   f"Valid: {is_valid_word}\n")

                    file.write('\n')
        except Exception as e:
            logging.error(f"Error saving summary stats to {output_file_path}: {e}", exc_info=True)

    def export_prediction_details_to_csv(self, predictions, prediction_method_name):
        split_type_str = f"_{self.split_type}" if self.split_type else ""
        csv_file_path = self.config.csv_dir / (
            f'{self.corpus_name}_{prediction_method_name}{split_type_str}_split'
            f'{self.config.split_config}_qrange{self.config.q_range[0]}-'
            f'{self.config.q_range[1]}_prediction.csv'
        )
        try:
            with csv_file_path.open('w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)

                writer.writerow([
                    'Tested_Word', 'Original_Word', 'Correct_Letter(s)', 
                    'Top1_Predicted_Letter', 'Top1_Confidence', 'Top1_Is_Valid', 'Top1_Is_Accurate',
                    'Top2_Predicted_Letter', 'Top2_Confidence', 'Top2_Is_Valid', 'Top2_Is_Accurate',
                    'Top3_Predicted_Letter', 'Top3_Confidence', 'Top3_Is_Valid', 'Top3_Is_Accurate',
                    'Correct_Letter_Rank', 'In_Training_Set'
                ])

                training_words_set = set(self.train_set)
                for mod_word, miss_letters, orig_word, top_preds, cor_letter_rank in predictions:
                    row = [mod_word, orig_word, ','.join(miss_letters)]

                    for predicted_letter, confidence in top_preds:
                        reconstructed_word = mod_word.replace('_', predicted_letter)
                        is_valid = 1 if reconstructed_word in self.all_words else 0
                        is_accurate = 1 if predicted_letter in miss_letters else 0

                        row.extend([predicted_letter, confidence, is_valid, is_accurate])

                    row.append(cor_letter_rank)
                    row.append(1 if orig_word in training_words_set else 0)

                    writer.writerow(row)
        except Exception as e:
            logging.error(f"Error exporting prediction details to CSV {csv_file_path}: {e}", exc_info=True)
