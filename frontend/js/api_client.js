/**
 * NetView API客户端
 * 负责与后端API的通信
 */

class APIClient {
    constructor() {
        this.baseURL = 'http://localhost:5000/api';
        this.timeout = 30000; // 30秒超时
    }

    /**
     * 发送HTTP请求的基础方法
     * @param {string} endpoint - API端点
     * @param {Object} options - 请求选项
     * @returns {Promise<Object>} 响应数据
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        try {
            console.log(`发送请求到: ${url}`, config);
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.timeout);
            
            const response = await fetch(url, {
                ...config,
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status} - ${response.statusText}`);
            }

            const data = await response.json();
            console.log('API响应:', data);
            return data;

        } catch (error) {
            console.error('API请求失败:', error);
            if (error.name === 'AbortError') {
                throw new Error('请求超时，请稍后重试');
            }
            throw error;
        }
    }

    /**
     * 解析PyTorch模型代码
     * @param {string} code - 模型代码
     * @param {Object} options - 解析选项
     * @returns {Promise<Object>} 解析结果
     */
    async parseModel(code, options = {}) {
        if (!code || !code.trim()) {
            throw new Error('模型代码不能为空');
        }

        return await this.request('/parse', {
            method: 'POST',
            body: JSON.stringify({
                code: code.trim(),
                options: {
                    show_params: options.showParams !== false,
                    show_shapes: options.showShapes !== false,
                    auto_layout: options.autoLayout !== false,
                    ...options
                }
            })
        });
    }

    /**
     * 获取图形数据
     * @param {string} sessionId - 会话ID
     * @returns {Promise<Object>} 图形数据
     */
    async getGraphData(sessionId) {
        if (!sessionId) {
            throw new Error('会话ID不能为空');
        }

        return await this.request(`/graph/${sessionId}`);
    }

    /**
     * 更新配置
     * @param {Object} config - 配置选项
     * @returns {Promise<Object>} 更新结果
     */
    async updateConfig(config) {
        return await this.request('/config', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    }

    /**
     * 导出图像
     * @param {string} format - 导出格式 (png, svg, pdf)
     * @param {Object} options - 导出选项
     * @returns {Promise<Blob>} 图像数据
     */
    async exportImage(format = 'png', options = {}) {
        const endpoint = `/export/${format}`;
        const response = await fetch(`${this.baseURL}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                width: options.width || 1200,
                height: options.height || 800,
                dpi: options.dpi || 300,
                ...options
            })
        });

        if (!response.ok) {
            throw new Error(`导出失败: ${response.status} - ${response.statusText}`);
        }

        return await response.blob();
    }

    /**
     * 获取解析状态
     * @param {string} sessionId - 会话ID
     * @returns {Promise<Object>} 状态信息
     */
    async getStatus(sessionId) {
        if (!sessionId) {
            throw new Error('会话ID不能为空');
        }

        return await this.request(`/status/${sessionId}`);
    }

    /**
     * 获取示例模型列表
     * @returns {Promise<Array>} 示例模型列表
     */
    async getExamples() {
        return await this.request('/examples');
    }

    /**
     * 获取示例模型代码
     * @param {string} exampleName - 示例名称
     * @returns {Promise<Object>} 示例代码
     */
    async getExampleCode(exampleName) {
        if (!exampleName) {
            throw new Error('示例名称不能为空');
        }

        return await this.request(`/examples/${exampleName}`);
    }

    /**
     * 健康检查
     * @returns {Promise<Object>} 服务状态
     */
    async healthCheck() {
        return await this.request('/health');
    }
}

// 错误处理工具类
class APIError extends Error {
    constructor(message, code = null, details = null) {
        super(message);
        this.name = 'APIError';
        this.code = code;
        this.details = details;
    }
}

// 创建全局API客户端实例
window.apiClient = new APIClient();

// 导出类和实例
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { APIClient, APIError };
}
