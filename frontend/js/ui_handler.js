/**
 * NetView UI处理器
 * 负责用户界面交互和事件处理
 */

class UIHandler {
    constructor() {
        this.currentSessionId = null;
        this.isProcessing = false;
        
        // DOM元素
        this.elements = {
            modelCode: null,
            parseBtn: null,
            clearBtn: null,
            exampleBtn: null,
            graphContainer: null,
            statusMessage: null
        };
        
        // 配置选项
        this.config = {
            showParams: true,
            showShapes: true,
            autoLayout: true
        };
        
        this.init();
    }

    /**
     * 初始化UI处理器
     */
    init() {
        // 等待DOM加载完成
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setup());
        } else {
            this.setup();
        }
    }

    /**
     * 设置UI元素和事件监听器
     */
    setup() {
        // 获取DOM元素
        this.elements = {
            modelCode: document.getElementById('model-code'),
            parseBtn: document.getElementById('parse-btn'),
            clearBtn: document.getElementById('clear-btn'),
            exampleBtn: document.getElementById('example-btn'),
            graphContainer: document.getElementById('graph-container'),
            statusMessage: document.getElementById('status-message')
        };

        // 添加事件监听器
        this.addEventListeners();
        
        // 设置初始状态
        this.setInitialState();
        
        // 检查后端连接
        this.checkBackendConnection();
        
        console.log('UI处理器初始化完成');
    }

    /**
     * 添加事件监听器
     */
    addEventListeners() {
        // 解析按钮点击事件
        if (this.elements.parseBtn) {
            this.elements.parseBtn.addEventListener('click', () => {
                this.handleParseClick();
            });
        }

        // 清空按钮点击事件
        if (this.elements.clearBtn) {
            this.elements.clearBtn.addEventListener('click', () => {
                this.handleClearClick();
            });
        }

        // 示例按钮点击事件
        if (this.elements.exampleBtn) {
            this.elements.exampleBtn.addEventListener('click', () => {
                this.handleExampleClick();
            });
        }

        // 代码输入框变化事件
        if (this.elements.modelCode) {
            this.elements.modelCode.addEventListener('input', () => {
                this.handleCodeChange();
            });
            
            // 支持Ctrl+Enter快捷键解析
            this.elements.modelCode.addEventListener('keydown', (event) => {
                if (event.ctrlKey && event.key === 'Enter') {
                    this.handleParseClick();
                }
            });
        }

        // 窗口大小变化事件
        window.addEventListener('resize', () => {
            this.handleResize();
        });

        // 配置选项变化事件
        this.addConfigEventListeners();
    }

    /**
     * 添加配置选项事件监听器
     */
    addConfigEventListeners() {
        const configOptions = document.querySelectorAll('.config-option input[type="checkbox"]');
        configOptions.forEach(option => {
            option.addEventListener('change', (event) => {
                const optionName = event.target.id.replace('-', '');
                this.config[this.toCamelCase(optionName)] = event.target.checked;
                console.log('配置更新:', this.config);
            });
        });
    }

    /**
     * 设置初始状态
     */
    setInitialState() {
        if (this.elements.graphContainer) {
            this.elements.graphContainer.innerHTML = `
                <div style="text-align: center; color: #666;">
                    <i class="fas fa-upload" style="font-size: 48px; margin-bottom: 16px;"></i>
                    <p>请输入PyTorch模型代码并点击"解析模型"</p>
                </div>
            `;
        }
        
        // 设置示例代码
        if (this.elements.modelCode && !this.elements.modelCode.value.trim()) {
            this.loadDefaultExample();
        }
    }

    /**
     * 加载默认示例
     */
    loadDefaultExample() {
        const defaultCode = `import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        x = x.view(-1, 64 * 8 * 8)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x`;

        if (this.elements.modelCode) {
            this.elements.modelCode.value = defaultCode;
        }
    }

    /**
     * 处理解析按钮点击
     */
    async handleParseClick() {
        if (this.isProcessing) {
            console.log('正在处理中，请稍候...');
            return;
        }

        const code = this.elements.modelCode?.value?.trim();
        if (!code) {
            this.showMessage('请输入模型代码', 'error');
            return;
        }

        try {
            this.setProcessingState(true);
            this.showMessage('正在解析模型...', 'info');
            
            // 调用API解析模型
            const result = await window.apiClient.parseModel(code, this.config);
            
            // 渲染可视化图形
            if (result.graph_data && window.visualizationEngine) {
                window.visualizationEngine.render(result.graph_data);
                this.showMessage('模型解析成功！', 'success');
                this.currentSessionId = result.session_id;
            } else {
                throw new Error('解析结果无效');
            }
            
        } catch (error) {
            console.error('解析失败:', error);
            this.showMessage(`解析失败: ${error.message}`, 'error');
            this.showErrorInGraph(error.message);
        } finally {
            this.setProcessingState(false);
        }
    }

    /**
     * 处理清空按钮点击
     */
    handleClearClick() {
        if (this.elements.modelCode) {
            this.elements.modelCode.value = '';
        }
        
        if (this.elements.graphContainer) {
            this.elements.graphContainer.classList.remove('has-content');
            this.setInitialState();
        }
        
        this.currentSessionId = null;
        this.clearMessage();
    }

    /**
     * 处理示例按钮点击
     */
    async handleExampleClick() {
        try {
            // 这里可以从后端获取示例列表
            const examples = [
                { name: 'SimpleCNN', label: '简单CNN' },
                { name: 'ResNet', label: 'ResNet网络' },
                { name: 'LSTM', label: 'LSTM网络' }
            ];
            
            // 简单起见，直接加载第一个示例
            this.loadDefaultExample();
            this.showMessage('已加载示例模型', 'info');
            
        } catch (error) {
            console.error('加载示例失败:', error);
            this.showMessage('加载示例失败', 'error');
        }
    }

    /**
     * 处理代码输入变化
     */
    handleCodeChange() {
        // 可以在这里添加代码验证逻辑
        const code = this.elements.modelCode?.value?.trim();
        if (this.elements.parseBtn) {
            this.elements.parseBtn.disabled = !code;
        }
    }

    /**
     * 处理窗口大小变化
     */
    handleResize() {
        // 重新初始化可视化引擎以适应新尺寸
        if (window.visualizationEngine && this.currentSessionId) {
            setTimeout(() => {
                if (typeof initVisualization === 'function') {
                    initVisualization();
                }
            }, 100);
        }
    }

    /**
     * 设置处理状态
     */
    setProcessingState(processing) {
        this.isProcessing = processing;
        
        if (this.elements.parseBtn) {
            this.elements.parseBtn.disabled = processing;
            this.elements.parseBtn.innerHTML = processing ? 
                '<span class="loading-spinner"></span> 解析中...' : '解析模型';
        }
        
        if (this.elements.graphContainer) {
            if (processing) {
                this.elements.graphContainer.classList.add('loading');
            } else {
                this.elements.graphContainer.classList.remove('loading');
            }
        }
    }

    /**
     * 显示状态消息
     */
    showMessage(message, type = 'info') {
        // 创建或更新状态消息元素
        let messageElement = document.querySelector('.status-message');
        
        if (!messageElement) {
            messageElement = document.createElement('div');
            messageElement.className = 'status-message';
            
            // 插入到合适的位置
            const inputSection = document.querySelector('.input-section');
            if (inputSection) {
                inputSection.insertBefore(messageElement, inputSection.firstChild);
            }
        }
        
        // 设置消息内容和样式
        messageElement.textContent = message;
        messageElement.className = `status-message status-${type}`;
        
        // 自动隐藏成功和信息消息
        if (type === 'success' || type === 'info') {
            setTimeout(() => {
                if (messageElement.parentNode) {
                    messageElement.remove();
                }
            }, 3000);
        }
    }

    /**
     * 清除状态消息
     */
    clearMessage() {
        const messageElement = document.querySelector('.status-message');
        if (messageElement) {
            messageElement.remove();
        }
    }

    /**
     * 在图形区域显示错误信息
     */
    showErrorInGraph(errorMessage) {
        if (this.elements.graphContainer) {
            this.elements.graphContainer.innerHTML = `
                <div style="text-align: center; color: #dc3545;">
                    <i class="fas fa-exclamation-triangle" style="font-size: 48px; margin-bottom: 16px;"></i>
                    <p><strong>解析失败</strong></p>
                    <p style="font-size: 14px; margin-top: 8px;">${errorMessage}</p>
                </div>
            `;
        }
    }

    /**
     * 检查后端连接
     */
    async checkBackendConnection() {
        try {
            await window.apiClient.healthCheck();
            console.log('后端连接正常');
        } catch (error) {
            console.warn('后端连接失败:', error);
            this.showMessage('无法连接到后端服务，某些功能可能不可用', 'error');
        }
    }

    /**
     * 工具函数：转换为驼峰命名
     */
    toCamelCase(str) {
        return str.replace(/-([a-z])/g, (g) => g[1].toUpperCase());
    }

    /**
     * 导出当前图形
     */
    async exportGraph(format = 'png') {
        if (!this.currentSessionId) {
            this.showMessage('请先解析模型', 'error');
            return;
        }

        try {
            if (format === 'svg' && window.visualizationEngine) {
                // 直接导出SVG
                window.visualizationEngine.exportSVG();
            } else {
                // 通过API导出其他格式
                const blob = await window.apiClient.exportImage(format);
                const url = URL.createObjectURL(blob);
                
                const link = document.createElement('a');
                link.href = url;
                link.download = `network_graph.${format}`;
                link.click();
                
                URL.revokeObjectURL(url);
            }
            
            this.showMessage('图形导出成功', 'success');
            
        } catch (error) {
            console.error('导出失败:', error);
            this.showMessage(`导出失败: ${error.message}`, 'error');
        }
    }
}

// 创建全局UI处理器实例
window.uiHandler = new UIHandler();

// 导出类
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { UIHandler };
}
