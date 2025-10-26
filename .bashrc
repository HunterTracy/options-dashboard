# Load up standard site-wide settings.
source /etc/bashrc

#remove duplicate entries from history
export HISTCONTROL=ignoreboth

# Show current git branch in prompt.
function parse_git_branch {
  git branch --no-color 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'
}
RED="\[\033[0;31m\]"
YELLOW="\[\033[0;33m\]"
GREEN="\[\033[0;32m\]"
LIGHT_GREEN="\[\033[1;32m\]"
RESET_COLOR="\[\033[0;0m\]"

PS1="$RESET_COLOR\$(date +%H:%M) \w$YELLOW \$(parse_git_branch)$LIGHT_GREEN\$ $RESET_COLOR"

# Load virtualenvwrapper
source virtualenvwrapper.sh &> /dev/null


# telegram
export TG_BOT_TOKEN="7941000306:AAGi9kyhkarTgHw9NyISLmG3VxQtsafp_7c"
export TG_CHAT_ID="8070619382"
# PostgreSQL
export DB_USER="Frogonacci"
export DB_PASSWORD="Hunter1303!"
export DB_HOST="sp500-db.csj66ggk01l4.us-east-1.rds.amazonaws.com"
export DB_NAME="options_db"

# Polygon.io API key
export POLYGON_API_KEY="o0yNtH7JGkD0GVzWLd0zriYUjeZ2ZoiD"

# Email (optional)
export EMAIL_ADDRESS="huntertracy1000@gmail.com"
export EMAIL_PASSWORD="ycwj ysag gxvk ulld"
