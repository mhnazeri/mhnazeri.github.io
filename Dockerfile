FROM docker.io/library/ruby:3.3-bookworm

LABEL org.opencontainers.image.authors="Amir Pourmand"

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential imagemagick \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=C.UTF-8 \
    LANGUAGE=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /srv/jekyll
COPY Gemfile Gemfile.lock ./
RUN gem install bundler -v 4.0.22 \
    && bundle install

WORKDIR /srv/jekyll