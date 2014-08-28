
.PHONY: push2github push2bitbucket push2all
push2all: push2bithub push2bitbucket
###
### GitHub
###
git-push:
	git push -u git@github.com:Daweek/dscudapkg1.4.2_FT2.git master

git-pull:
	git pull -u git@github.com:Daweek/dscudapkg1.4.2_FT2.git master
###
### Bitbucket
###
bitbucket-push:
	git push -u git@bitbucket.org:m_oikawa/dscudapkg1.4.2_FT.git --all
