# Tests

## Overview

Basic validation tests to ensure configuration and module initialization work correctly.

## Running Tests

```bash
python tests/test_config.py
```

## Test Coverage

### test_config.py
- Configuration file loading and validation
- Module initialization with config parameters
- Type safety verification
- Parameter bounds checking

## Expected Output

```
[PASS] Config validation
[PASS] Module initialization

[SUCCESS] All tests passed
```

## Adding Tests

To add new tests:
1. Create a new test file in this directory
2. Import required modules
3. Write test functions with descriptive names
4. Use assertions to verify behavior
5. Print clear pass/fail messages

## Notes

These are basic smoke tests, not comprehensive unit tests. They verify:
- Config loads without errors
- Modules initialize correctly
- Type casting works as expected
- No obvious regressions

For production use, consider adding:
- Unit tests for each module
- Integration tests for full pipeline
- Property-based tests for edge cases
- Performance benchmarks
