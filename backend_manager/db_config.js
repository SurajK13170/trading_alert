const mysql = require('mysql2/promise');
const redis = require('redis');
require('dotenv').config();

const cleanEnv = (name, defaultValue) => {
    const value = process.env[name] ?? defaultValue;
    return typeof value === 'string' ? value.split('  #')[0].trim() : value;
};

const mysqlConfig = {
    host: cleanEnv('MYSQL_HOST', 'localhost'),
    user: cleanEnv('MYSQL_USER', 'root'),
    password: cleanEnv('MYSQL_PASSWORD', ''),
    database: cleanEnv('MYSQL_DATABASE') || cleanEnv('MYSQL_DB', 'quant_db')
};

// MySQL Connection Pool
const mysqlPool = mysql.createPool({
    ...mysqlConfig,
    waitForConnections: true,
    connectionLimit: 10
});

// Redis Client
const redisClient = redis.createClient({
    url: cleanEnv('REDIS_URL', 'redis://localhost:6379')
});

redisClient.on('error', (err) => console.log('Redis Client Error', err));

const connectDB = async () => {
    try {
        if (!redisClient.isOpen) {
            await redisClient.connect();
        }
        console.log("Redis Connected");
    } catch (err) {
        console.warn("Redis unavailable; continuing without Redis-backed features.", err.message || err);
    }

    try {
        // Create database if it doesn't exist. If MySQL is not installed or not
        // reachable, continue anyway so the backend can still boot for partial
        // deployments and UI-only validation.
        const tempConnection = await mysql.createConnection({
            host: mysqlConfig.host,
            user: mysqlConfig.user,
            password: mysqlConfig.password,
            connectTimeout: 2000,
        });
        await tempConnection.execute(`CREATE DATABASE IF NOT EXISTS \`${mysqlConfig.database}\``);
        await tempConnection.end();

        const connection = await mysqlPool.getConnection();
        console.log("MySQL Connected");
        connection.release();
    } catch (err) {
        console.warn("MySQL unavailable; continuing without database-backed features.", err.message || err);
    }
};

module.exports = { mysqlPool, redisClient, connectDB };
