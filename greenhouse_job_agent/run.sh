#!/bin/bash
#
# Greenhouse Job Application Agent - Run Script
#
# Usage:
#   ./run.sh              # Run in continuous mode (polls every 15 min)
#   ./run.sh --once       # Run once and exit
#   ./run.sh --setup      # Interactive setup wizard
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  Greenhouse Job Application Agent${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo
}

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }

# Check if .env file exists, if not create from example
check_env() {
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            print_warning ".env file not found, creating from .env.example"
            cp .env.example .env
            print_warning "Please edit .env with your API keys before running"
            return 1
        else
            print_error ".env.example not found"
            return 1
        fi
    fi
    return 0
}

# Load environment variables from .env
load_env() {
    if [ -f ".env" ]; then
        set -a
        source .env
        set +a
        print_success "Loaded configuration from .env"
    fi
}

# Check required dependencies
check_dependencies() {
    echo "Checking dependencies..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    print_success "Python 3 found: $(python3 --version)"

    # Check if requirements are installed
    if ! python3 -c "import langchain_openai" &> /dev/null; then
        print_warning "Dependencies not installed. Installing..."
        pip3 install -r requirements.txt
        print_success "Dependencies installed"
    else
        print_success "Dependencies already installed"
    fi
}

# Validate configuration
validate_config() {
    local errors=0

    echo "Validating configuration..."

    # Check OpenAI API key
    if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "sk-your-openai-api-key-here" ]; then
        print_error "OPENAI_API_KEY not set or invalid"
        errors=$((errors + 1))
    else
        print_success "OPENAI_API_KEY configured"
    fi

    # Check Greenhouse board token
    if [ -z "$GH_BOARD_TOKEN" ] || [ "$GH_BOARD_TOKEN" = "examplecompany" ]; then
        print_error "GH_BOARD_TOKEN not set or invalid"
        echo "       Examples: anthropic, stripe, notion, airbnb"
        errors=$((errors + 1))
    else
        print_success "GH_BOARD_TOKEN: $GH_BOARD_TOKEN"
    fi

    # Check resume file
    if [ -f "resume.md" ]; then
        print_success "Resume found: resume.md ($(wc -c < resume.md) bytes)"
    elif [ -n "$RESUME_FILE" ] && [ -f "$RESUME_FILE" ]; then
        print_success "Resume found: $RESUME_FILE"
    else
        print_warning "No resume.md found - will use default sample"
    fi

    # Show user info
    echo
    echo "User Configuration:"
    echo "  Name:  ${USER_FIRST_NAME:-Not set} ${USER_LAST_NAME:-Not set}"
    echo "  Email: ${USER_EMAIL:-Not set}"
    echo "  Phone: ${USER_PHONE:-Not set}"

    # Show job filters
    echo
    echo "Job Filters:"
    echo "  Query:     ${JOB_QUERY:-None (all jobs)}"
    echo "  Locations: ${JOB_LOCATIONS:-None (all locations)}"

    if [ $errors -gt 0 ]; then
        echo
        print_error "Configuration has $errors error(s). Please fix before running."
        echo "Edit .env file: nano .env"
        return 1
    fi

    return 0
}

# Interactive setup wizard
setup_wizard() {
    print_header
    echo "Interactive Setup Wizard"
    echo

    # Create .env if doesn't exist
    if [ ! -f ".env" ]; then
        cp .env.example .env 2>/dev/null || touch .env
    fi

    # OpenAI API Key
    echo -e "${YELLOW}1. OpenAI API Key${NC}"
    echo "   Get one at: https://platform.openai.com/api-keys"
    read -p "   Enter your OpenAI API key: " api_key
    if [ -n "$api_key" ]; then
        sed -i "s/^OPENAI_API_KEY=.*/OPENAI_API_KEY=$api_key/" .env 2>/dev/null || \
            echo "OPENAI_API_KEY=$api_key" >> .env
        print_success "OpenAI API key saved"
    fi
    echo

    # Greenhouse Board Token
    echo -e "${YELLOW}2. Greenhouse Board Token${NC}"
    echo "   Find this in company careers URLs: boards.greenhouse.io/{token}"
    echo "   Examples: anthropic, stripe, notion, airbnb, figma"
    read -p "   Enter board token: " board_token
    if [ -n "$board_token" ]; then
        sed -i "s/^GH_BOARD_TOKEN=.*/GH_BOARD_TOKEN=$board_token/" .env 2>/dev/null || \
            echo "GH_BOARD_TOKEN=$board_token" >> .env
        print_success "Board token saved: $board_token"
    fi
    echo

    # User Info
    echo -e "${YELLOW}3. Your Contact Info${NC}"
    read -p "   First name: " first_name
    read -p "   Last name: " last_name
    read -p "   Email: " email
    read -p "   Phone (optional): " phone

    [ -n "$first_name" ] && sed -i "s/^USER_FIRST_NAME=.*/USER_FIRST_NAME=$first_name/" .env
    [ -n "$last_name" ] && sed -i "s/^USER_LAST_NAME=.*/USER_LAST_NAME=$last_name/" .env
    [ -n "$email" ] && sed -i "s/^USER_EMAIL=.*/USER_EMAIL=$email/" .env
    [ -n "$phone" ] && sed -i "s/^USER_PHONE=.*/USER_PHONE=$phone/" .env
    print_success "Contact info saved"
    echo

    # Job Filters
    echo -e "${YELLOW}4. Job Filters (optional)${NC}"
    read -p "   Job title keyword (e.g., 'product', 'engineer'): " job_query
    read -p "   Locations (comma-separated, e.g., 'Remote,San Francisco'): " locations

    [ -n "$job_query" ] && sed -i "s/^JOB_QUERY=.*/JOB_QUERY=$job_query/" .env
    [ -n "$locations" ] && sed -i "s/^JOB_LOCATIONS=.*/JOB_LOCATIONS=$locations/" .env
    echo

    print_success "Setup complete!"
    echo
    echo "Your configuration has been saved to .env"
    echo "Run './run.sh --once' to test, or './run.sh' for continuous mode"
}

# Main run function
run_agent() {
    local run_once=${1:-false}

    print_header

    check_dependencies
    echo

    load_env

    if ! validate_config; then
        exit 1
    fi

    echo
    echo -e "${GREEN}Starting agent...${NC}"
    echo

    if [ "$run_once" = true ]; then
        export RUN_ONCE=true
        python3 agent.py
    else
        echo "Running in continuous mode (Ctrl+C to stop)"
        echo "Poll interval: ${POLL_INTERVAL:-900} seconds"
        echo
        python3 agent.py
    fi
}

# Show help
show_help() {
    print_header
    echo "Usage: ./run.sh [OPTIONS]"
    echo
    echo "Options:"
    echo "  --once      Run once and exit (good for testing)"
    echo "  --setup     Interactive setup wizard"
    echo "  --check     Check configuration without running"
    echo "  --help      Show this help message"
    echo
    echo "Examples:"
    echo "  ./run.sh --setup     # First time setup"
    echo "  ./run.sh --once      # Test run (apply to jobs once)"
    echo "  ./run.sh             # Continuous mode (polls every 15 min)"
    echo
    echo "Configuration:"
    echo "  Edit .env file to customize settings"
    echo "  Edit resume.md to use your own resume"
}

# Parse command line arguments
case "${1:-}" in
    --setup)
        setup_wizard
        ;;
    --once)
        run_agent true
        ;;
    --check)
        print_header
        load_env
        validate_config
        ;;
    --help|-h)
        show_help
        ;;
    "")
        run_agent false
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
