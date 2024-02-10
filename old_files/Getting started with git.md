# Getting Started
0. Accept the GitHub invite first
1. Install GIT from here:
>*https://git-scm.com/downloads*
2. Open VSCode -> open terminal (CTRL + J) -> open Git Bash from the drop down arrow on the top right of terminal
3. Go to your working directory, preferably an empty folder
4. To clone our repository, type
>`git clone https://github.com/jblagana/eee-capstone.git`
5. DONE! Feel free to add files, edit contents, etc.
6. To see your changes, type
>`git status`
7. To save specific changes, type
>`git add <file path from git status>`

Alternatively, to save all your changes, type:
>`git add .`

8. To commit your changes, type
>`git commit -m "message here (what changes were done)"`

9. To push your changes to GitHub for the first time, type
>`git push -u origin master`

For succeeding pushes, just type
>`git push`

## Notes
> 1. The most common commands you will use all the time are `git status`, `git add`, `git commit`, `git push`.<br>
> 2. By convention, we add a new branch in GitHub where we will make our personal changes to prevent altering the main/master branch accidentally.<br>
> For simplicity, we just ignore branches. <br>
> 3. To **SEE** the other users' changes in our local machines, type `git fetch`.<br>
> 4. To **SAVE** the other users' changes in our local machines, type `git pull`.