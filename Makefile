
.PHONY: push2github push2bitbucket push2all
push2all: push2bithub push2bitbucket

push2github:
#	git push -u https://github.com/Daweek/dscudapkg1.4.2_FT2.git master
	git push -u git@github.com:Daweek/dscudapkg1.4.2_FT2.git master

push2bitbucket:
	git push -u git@bitbucket.org:m_oikawa/dscudapkg1.4.2_FT.git --all
