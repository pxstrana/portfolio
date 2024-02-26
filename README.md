# Portfolio using Jekyll and Bulma clean theme

Hello! I've created this portfolio using **Jekyll**, a tool for creating static websites like blogs, and **Bulma** to create a responsive web. For this, I've used the template from ```chrisrhymes/bulma-clean-theme```.

This is my first experience creating a blog. Initially, I aim to cover topics related to data science and AI, and to try to make some of these accessible to people with no prior knowledge. 

I'm no expert by any means, so feel free to comment with any errors or critiques :)

## Start the server

Run ```bundle exec jekyll serve``` and browse ```http://localhost:4000```
If you want to see the changes on the fly make sure to have this line in ```_config.yml```: ````livereload: true````

## Upload to Github Pages

Once you are ready, you can upload to Github Pages with the following config in ````_config.yml````:
- remote_theme: your_name/your_repo
- baseurl: "/your_repo"
- url: "https://your_name.github.io/

😉
