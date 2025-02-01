import logging
import os
from datetime import datetime

def setup_logging(agent_name=None):
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Generate timestamp for log file name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create logger
    logger = logging.getLogger(agent_name or __name__)
    
    if not logger.handlers:  # Only add handlers if they don't exist
        logger.setLevel(logging.INFO)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create handlers
        # Main log file
        main_handler = logging.FileHandler(
            f'logs/svg_generator_{timestamp}.txt',
            encoding='utf-8'
        )
        main_handler.setFormatter(detailed_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(detailed_formatter)
        
        # Agent-specific log file (if agent_name is provided)
        if agent_name:
            agent_handler = logging.FileHandler(
                f'logs/{agent_name}_{timestamp}.txt',
                encoding='utf-8'
            )
            agent_handler.setFormatter(detailed_formatter)
            logger.addHandler(agent_handler)
        
        # Add handlers
        logger.addHandler(main_handler)
        logger.addHandler(console_handler)
        
        # Log initialization
        logger.info(f"Logging initialized for {agent_name or 'main'}")
        if agent_name:
            logger.info(f"Agent-specific log file created: {agent_name}_{timestamp}.txt")
    
    return logger 