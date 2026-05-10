FROM node:20-alpine

# bcrypt requires native compilation (node-gyp)
RUN apk add --no-cache python3 make g++

WORKDIR /app

COPY package.json ./
RUN npm install

COPY index.js ./
COPY public/ ./public/

# /data is mounted from the host via docker-compose volume
ENV ISD_DB_PATH=/data/host.db
ENV PORT=3000
ENV HTTPS_PORT=3443

# HTTP (redirects to HTTPS when TLS is configured)
EXPOSE 3000
# HTTPS
EXPOSE 3443

CMD ["node", "index.js"]