FROM node:22-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY src ./src
COPY index.html vite.config.ts tsconfig.json server.ts ./
EXPOSE 3000
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
