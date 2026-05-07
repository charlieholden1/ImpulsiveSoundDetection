FROM node:20-alpine

WORKDIR /app

COPY package.json ./
RUN npm install

COPY index.js ./
COPY public/ ./public/

ENV ISD_DB_PATH=/data/host.db

EXPOSE 3000
CMD ["node", "index.js"]