#!/usr/bin/env bash

set -e

# ANSI Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}  CLIRAG One-Click Installer (Linux/Mac)  ${NC}"
echo -e "${CYAN}==========================================${NC}"

# 1. Check for Python
echo -e "\n${YELLOW}[1/4] Checking minimum requirements...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed or not in PATH. Please install Python 3.8+.${NC}"
    exit 1
fi
python3 --version | xargs echo -e "${GREEN}Detected:${NC}"

# 2. Check Repository
echo -e "\n${YELLOW}[2/4] Verifying CLIRAG repository...${NC}"
if [ ! -f "setup.py" ]; then
    echo -e "${YELLOW}Warning: setup.py not found. Assuming external run.${NC}"
    # git clone https://github.com/yourusername/clirag.git
    # cd clirag
else
    echo -e "${GREEN}CLIRAG source validated.${NC}"
fi

# 3. Create Virtual Environment
echo -e "\n${YELLOW}[3/4] Creating an isolated Python Virtual Environment (venv)...${NC}"
if [ ! -d "env" ]; then
    python3 -m venv env
    echo -e "${GREEN}Virtual environment 'env' created.${NC}"
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi

# 4. Install CLIRAG
echo -e "\n${YELLOW}[4/4] Installing CLIRAG dependencies and binding 'clirag' CLI command...${NC}"
source env/bin/activate
pip install --upgrade pip
pip install -e .

echo -e "\n${CYAN}==========================================${NC}"
echo -e "${GREEN}  ✅ INSTALLATION COMPLETE  ${NC}"
echo -e "${CYAN}==========================================${NC}"
echo -e "\nTo use CLIRAG, you must activate the virtual environment and type 'clirag':"
echo -e "  $ source env/bin/activate"
echo -e "  $ clirag --help"
echo -e "  $ clirag ingest ./path/to/file.pdf\n"
