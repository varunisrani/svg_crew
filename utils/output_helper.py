def get_output_str(output, max_length=None):
    """Convert CrewOutput or any other output to string safely"""
    # Convert to string, handling CrewOutput objects
    output_str = str(output.raw_output if hasattr(output, 'raw_output') else output)
    
    # Truncate if max_length is specified
    if max_length and len(output_str) > max_length:
        return f"{output_str[:max_length]}..."
    
    return output_str 