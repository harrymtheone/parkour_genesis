from dataclasses import dataclass, MISSING


def configclass(cls):
    """
    Decorator to create configuration classes with validation.
    
    This decorator:
    1. Applies the dataclass decorator
    2. Adds validation for MISSING required parameters
    3. Provides helpful error messages for missing configuration
    
    Usage:
        @configclass
        class MyConfig:
            required_param: str = MISSING
            optional_param: int = 42
    
    Args:
        cls: The class to be decorated
        
    Returns:
        The decorated class with validation capabilities
    """
    # Apply dataclass decorator
    cls = dataclass(cls)

    # Store original __post_init__ if it exists
    original_post_init = getattr(cls, '__post_init__', None)

    def __post_init__(self):
        """Post-initialization hook that validates configuration."""
        # Validate configuration
        self.validate()

        # Call original __post_init__ if it existed
        if original_post_init:
            original_post_init(self)

    def validate(self):
        """
        Validate configuration and check for missing required parameters.
        
        Raises:
            ValueError: If any required parameters are missing (marked with MISSING)
        """
        missing_params = []

        # Check for MISSING values in all fields
        for field_name, field_obj in self.__dataclass_fields__.items():
            field_value = getattr(self, field_name)
            if field_value is MISSING:
                missing_params.append(field_name)

        if missing_params:
            raise ValueError(
                f"Required configuration parameters are missing in {self.__class__.__name__}: {missing_params}"
            )

    # Add methods to the class
    cls.__post_init__ = __post_init__
    cls.validate = validate
    return cls


# Re-export MISSING for convenience
__all__ = ['configclass', 'MISSING']
