mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"c.berragan@liverpool.ac.uk\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n
