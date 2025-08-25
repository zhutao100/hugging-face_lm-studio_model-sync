# LM Studio - Hugging Face Model Manager

A command-line utility to seamlessly manage and synchronize machine learning models between your Hugging Face cache and LM Studio. This tool simplifies the process of importing, exporting, and maintaining models across both platforms.

## Key Features

- **Bidirectional Synchronization:** Effortlessly sync models in both directions—from Hugging Face to LM Studio and from LM Studio to Hugging Face.
- **Interactive Model Selection:** A user-friendly interface to browse and select models for synchronization. It works on both UNIX-like systems and Windows.
- **Space-Efficient:** Uses symbolic links to import and export models, saving significant disk space by avoiding file duplication. Alternatively, use move mode to physically transfer files.
- **Automatic Cleanup:** In move mode, automatically cleans up source directories when they become empty.
- **Smart Model Discovery:** Automatically detects models in your Hugging Face and LM Studio directories.
- **Flexible Configuration:** Customize cache directories using command-line arguments or environment variables.
- **Dry Run Mode:** Preview all file operations without making any actual changes to your system.
- **Verbose Logging:** Enable detailed logging for debugging purposes.

## Prerequisites

- Python 3.x
- LM Studio installed
- Hugging Face models downloaded locally

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/ivanfioravanti/lmstudio_hf.git
    cd lmstudio_hf
    ```

## Usage

Run the script from your terminal:

```bash
python lmstudio_hf.py
```

### Command-Line Options

- `--dry-run`: Perform a dry run to preview operations.
- `--hf-cache-dir`: Specify a custom Hugging Face cache directory.
- `--lm-studio-dir`: Specify a custom LM Studio models directory.
- `--verbose`: Enable verbose (debug) logging.
- `--mode`: Specify the sync mode: 'symlink' (default) or 'move'.

### Environment Variables

- `HF_HOME`: Custom Hugging Face cache location.
- `XDG_CACHE_HOME`: Base directory for the cache on Linux.
- `LMSTUDIO_HOME`: Custom LM Studio models directory.

## How It Works

The script scans both the Hugging Face and LM Studio model directories to identify all available models. It then presents an interactive list for you to select which models to sync.

### Sync Modes

The tool supports two synchronization modes that can be selected using the `--mode` argument:

1. **Symlink Mode (Default)**: Creates symbolic links between the directories, saving disk space by avoiding file duplication. This is the default behavior and maintains the original space-efficient approach.

2. **Move Mode**: Physically moves files between directories, which actually transfers the file data. This mode uses more disk space during the transfer process but results in completely independent copies of the models. In move mode, the tool will also attempt to clean up the source directories:
   - When moving from Hugging Face to LM Studio: The Hugging Face model directory will be removed if no other blob files exist
   - When moving from LM Studio to Hugging Face: The LM Studio model directory will be removed if no files remain

### Hugging Face to LM Studio

When you select a model from your Hugging Face cache, the script will either:

- Create symbolic links to it in your LM Studio models directory (symlink mode)
- Move the files to your LM Studio models directory (move mode)

If a model is already present, you can choose to remove it.

### LM Studio to Hugging Face

When you select a local model from LM Studio, the script fetches its metadata from the Hugging Face Hub, creates the necessary directory structure in your Hugging Face cache, and:

- Creates symbolic links to the model files in the Hugging Face blobs directory (symlink mode)
- Moves the model files to the Hugging Face blobs directory (move mode)

In both modes, the snapshot files are always symbolic links to the blob files to maintain the Hugging Face directory structure.

If a model is already present in the Hugging Face cache, you can choose to remove it and re-sync.

## Running Tests

The project includes a test suite to ensure functionality works correctly.

```bash
# Run all tests
python run_tests.py
```
## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
