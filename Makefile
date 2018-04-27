
VERSION := $(shell echo `cat ./version/version.txt`-`git log -1 --pretty=format:%h`)
# tann = trivial artificial neural network
SERVICE_NAME := tann

all: clean build run

clean:
	rm -rf bin/*

build:
	CGO_ENABLED=0 go build -ldflags '-X github.com/marcospedreiro/$(SERVICE_NAME)/version.VERSION=$(VERSION)' -o bin/$(SERVICE_NAME)_$(VERSION)

run:
	./bin/$(SERVICE_NAME)_$(VERSION)

release: semver linux darwin windows

linux:
	env GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -ldflags '-X github.com/marcospedreiro/$(SERVICE_NAME)/version.VERSION=$(VERSION)' -o bin/$(SERVICE_NAME)_linux_amd64_$(VERSION)

darwin:
	env GOOS=darwin GOARCH=amd64 CGO_ENABLED=0 go build -ldflags '-X github.com/marcospedreiro/$(SERVICE_NAME)/version.VERSION=$(VERSION)' -o bin/$(SERVICE_NAME)_darwin_amd64_$(VERSION)

windows:
	env GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -ldflags '-X github.com/marcospedreiro/$(SERVICE_NAME)/version.VERSION=$(VERSION)' -o bin/$(SERVICE_NAME)_windows_amd64_$(VERSION).exe

increment_version:
	./version/increment.sh

.PHONY: build run clean release linux darwin windows semver
