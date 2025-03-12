"""El módulo main.py actúa como punto de entrada principal del sistema de detección
de fraude. Su función es orquestar todo el flujo de trabajo manteniendo un alto nivel
de abstracción."""

from pathlib import Path
from config.settings import CONFIG
from model_training.model_factory import ModelFactory
from utils.logger import setup_logger
from data_preparation.pipeline import DataPipeline


def main():
    """Main execution pipeline for fraud detection system."""
    # Setup logging
    logger = setup_logger()
    logger.info("Starting fraud detection pipeline")

    try:
        # Create data pipeline
        data_pipeline = DataPipeline()

        # Process data through the pipeline
        train_data, val_data, test_data = data_pipeline.process()
        logger.info("Data processing complete")

        # Train and evaluate models using factory pattern
        model_factory = ModelFactory(train_data, val_data, test_data)

        # Get list of models to train from configuration
        models_to_train = CONFIG["models"]["enabled"]

        # Train each enabled model
        results = {}
        for model_name in models_to_train:
            logger.info("Training model: %s", model_name)
            model = model_factory.create_model(model_name)
            model_results = model.train_and_evaluate()
            results[model_name] = model_results

        # Export results
        export_results(results)
        logger.info("Fraud detection pipeline completed successfully")

    except Exception as e:
        logger.error("Error in fraud detection pipeline: %s", e)
        raise


def export_results(results):
    """Export model results to the specified output directory."""
    output_dir = Path(CONFIG["output"]["directory"])
    output_dir.mkdir(exist_ok=True, parents=True)

    for model_name, model_results in results.items():
        result_path = output_dir / f"{model_name}_results.csv"
        model_results.to_csv(result_path, index=False)


if __name__ == "__main__":
    main()
