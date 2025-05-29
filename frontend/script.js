// 状态管理类
class AppState {
    constructor() {
        this.theme = localStorage.getItem('theme') || 'light';
        this.currentTask = null;
        this.networkInstance = null;
        this.isAnalyzing = false;
        this.analysisProgress = 0;
        
        this.initializeTheme();
    }
    
    initializeTheme() {
        document.documentElement.setAttribute('data-theme', this.theme);
        this.updateThemeButton();
    }
    
    toggleTheme() {
        this.theme = this.theme === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', this.theme);
        localStorage.setItem('theme', this.theme);
        this.updateThemeButton();
        console.log('主题已切换为:', this.theme);
    }
    
    updateThemeButton() {
        const themeButton = document.getElementById('theme-toggle');
        if (themeButton) {
            themeButton.textContent = this.theme === 'light' ? '🌙 暗色主题' : '☀️ 亮色主题';
        }
    }
    
    setAnalyzing(analyzing) {
        this.isAnalyzing = analyzing;
        this.updateAnalyzeButton();
    }
    
    updateAnalyzeButton() {
        const button = document.getElementById('analyze-button');
        const text = document.getElementById('analyze-text');
        if (button && text) {
            button.disabled = this.isAnalyzing;
            text.textContent = this.isAnalyzing ? '⏳ 分析中...' : '🚀 开始分析';
        }
    }
    
    updateProgress(progress, stage = '') {
        this.analysisProgress = Math.min(100, Math.max(0, progress));
        const progressBar = document.getElementById('loading-progress-bar');
        const loadingText = document.querySelector('.loading-text');
        
        if (progressBar) {
            progressBar.style.width = `${this.analysisProgress}%`;
        }
        
        if (loadingText && stage) {
            loadingText.textContent = `正在${stage}... (${Math.round(this.analysisProgress)}%)`;
        }
    }
}

// 通知系统
class NotificationManager {
    constructor() {
        this.container = document.getElementById('notification');
        this.textElement = document.getElementById('notification-text');
    }
    
    show(message, type = 'success', duration = 3000) {
        if (!this.container || !this.textElement) return;
        
        this.textElement.textContent = message;
        this.container.className = `visible ${type}`;
        this.container.style.display = 'block';
        
        // 添加动画效果
        this.container.style.animation = 'slideInRight 0.3s ease-out';
        
        console.log(`[通知] ${type.toUpperCase()}: ${message}`);
        
        setTimeout(() => {
            this.hide();
        }, duration);
    }
    
    hide() {
        if (!this.container) return;
        
        this.container.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => {
            this.container.style.display = 'none';
            this.container.className = 'hidden';
        }, 300);
    }
}

// 网络图增强管理器
class NetworkGraphManager {
    constructor() {
        this.network = null;
        this.container = document.getElementById('network-graph');
        this.isFullscreen = false;
    }
    
    destroy() {
        if (this.network) {
            this.network.destroy();
            this.network = null;
        }
    }
    
    render(nodesData, edgesData) {
        if (!nodesData || !edgesData) {
            this.showError('无法加载模型图数据');
            return;
        }

        console.log('渲染网络图，节点数:', nodesData.length, '边数:', edgesData.length);

        // 清空旧图
        this.destroy();
        this.container.innerHTML = '';

        // 处理节点数据
        const visNodes = nodesData.map(node => ({
            id: node.id,
            label: this.formatNodeLabel(node),
            title: this.formatNodeTooltip(node),
            shape: 'box',
            color: this.getNodeColor(node.type),
            font: { 
                size: 12, 
                face: 'arial',
                color: '#333'
            },
            borderWidth: 2,
            margin: 10
        }));

        // 处理边数据
        const visEdges = edgesData.map((edge, index) => ({
            from: edge.source,
            to: edge.target,
            arrows: 'to',
            label: this.getEdgeLabel(edge.connection_type),
            id: `edge_${index}`,
            font: { 
                size: 9, 
                color: '#666',
                strokeWidth: 2,
                strokeColor: 'white'
            },
            smooth: {
                type: 'cubicBezier',
                forceDirection: 'vertical',
                roundness: 0.2 + (index * 0.1) % 0.4
            },
            width: 2
        }));

        // 增强连接
        const enhancedEdges = this.enhanceConnections(visNodes, visEdges);

        const data = {
            nodes: new vis.DataSet(visNodes),
            edges: new vis.DataSet(enhancedEdges),
        };

        const options = {
            layout: {
                hierarchical: {
                    direction: "UD",
                    sortMethod: "directed",
                    levelSeparation: 150,
                    nodeSpacing: 100,
                    treeSpacing: 120,
                    blockShifting: true,
                    edgeMinimization: true,
                    parentCentralization: true
                }
            },
            physics: {
                enabled: false
            },
            interaction: {
                dragNodes: true,
                dragView: true,
                zoomView: true,
                hover: true,
                selectConnectedEdges: true,
                tooltipDelay: 200
            },
            nodes: {
                font: {
                    size: 12,
                    face: 'arial',
                    color: '#333'
                },
                margin: 10,
                widthConstraint: { maximum: 160 },
                heightConstraint: { minimum: 40 },
                borderWidth: 2,
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.1)',
                    size: 5,
                    x: 2,
                    y: 2
                }
            },
            edges: {
                font: {
                    size: 9,
                    align: 'middle',
                    color: '#666',
                    strokeWidth: 2,
                    strokeColor: 'white'
                },
                color: {
                    color: '#848484',
                    highlight: '#2563eb',
                    hover: '#2563eb'
                },
                width: 2,
                smooth: {
                    type: 'cubicBezier',
                    forceDirection: 'vertical',
                    roundness: 0.3
                },
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.05)',
                    size: 3,
                    x: 1,
                    y: 1
                }
            }
        };

        try {
            this.network = new vis.Network(this.container, data, options);
            
            // 添加事件监听
            this.network.on('selectNode', (params) => {
                console.log('选中节点:', params.nodes);
                // 可以在这里添加节点选中的处理逻辑
            });
            
            this.network.on('hoverNode', (params) => {
                this.container.style.cursor = 'pointer';
            });
            
            this.network.on('blurNode', (params) => {
                this.container.style.cursor = 'default';
            });
            
            console.log('网络图渲染完成');
            
            // 适应视图
            setTimeout(() => {
                this.network.fit({
                    animation: {
                        duration: 1000,
                        easingFunction: 'easeInOutQuad'
                    }
                });
            }, 100);
            
        } catch (error) {
            console.error('网络图渲染失败:', error);
            this.showError('网络图渲染失败: ' + error.message);
        }
    }
    
    formatNodeLabel(node) {
        const label = node.label || node.type || 'Unknown';
        const params = node.num_parameters;
        if (params && params > 0) {
            return `${label}\n(${this.formatNumber(params)}参数)`;
        }
        return `${label}\n(${node.id})`;
    }
    
    formatNodeTooltip(node) {
        let tooltip = `<strong>类型:</strong> ${node.type}<br>`;
        tooltip += `<strong>ID:</strong> ${node.id}<br>`;
        
        if (node.num_parameters) {
            tooltip += `<strong>参数:</strong> ${this.formatNumber(node.num_parameters)}<br>`;
        }
        
        if (node.output_shape) {
            tooltip += `<strong>输出形状:</strong> ${this.formatShape(node.output_shape)}<br>`;
        }
        
        if (node.input_shape) {
            tooltip += `<strong>输入形状:</strong> ${this.formatShape(node.input_shape)}`;
        }
        
        return tooltip;
    }
    
    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }
    
    formatShape(shape) {
        if (!shape || shape === 'N/A') return 'Dynamic';
        if (Array.isArray(shape)) {
            return shape.join(' × ');
        }
        return String(shape);
    }
    
    getEdgeLabel(connectionType) {
        if (!connectionType) return '';
        const labelMap = {
            'sequential': '',
            'reshape': 'view',
            'residual': '+',
            'attention': 'attn',
            'branch': 'split',
            'merge': 'concat'
        };
        return labelMap[connectionType] || connectionType;
    }
    
    enhanceConnections(nodes, edges) {
        const enhancedEdges = [...edges];
        
        // 查找卷积层和线性层之间的隐式连接
        const convNodes = nodes.filter(node => 
            node.id.includes('conv') || 
            node.label.toLowerCase().includes('conv')
        );
        const linearNodes = nodes.filter(node => 
            node.id.includes('fc') || 
            node.label.toLowerCase().includes('linear')
        );
        
        if (convNodes.length > 0 && linearNodes.length > 0) {
            const lastConv = convNodes[convNodes.length - 1];
            const firstLinear = linearNodes[0];
            
            const connectionExists = edges.some(edge => 
                edge.from === lastConv.id && edge.to === firstLinear.id
            );
            
            if (!connectionExists) {
                enhancedEdges.push({
                    from: lastConv.id,
                    to: firstLinear.id,
                    arrows: 'to',
                    label: 'flatten',
                    id: 'auto_flatten',
                    font: { 
                        size: 8, 
                        color: '#999',
                        strokeWidth: 2,
                        strokeColor: 'white'
                    },
                    color: { color: '#999' },
                    dashes: [5, 5],
                    smooth: {
                        type: 'cubicBezier',
                        forceDirection: 'vertical',
                        roundness: 0.5
                    }
                });
            }
        }
        
        return enhancedEdges;
    }
    
    getNodeColor(nodeType) {
        const typeLower = nodeType.toLowerCase();
        
        if (typeLower.includes('conv')) 
            return { background: '#FFC107', border: '#FF8F00', highlight: { background: '#FFD54F', border: '#FF8F00' } };
        if (typeLower.includes('pool')) 
            return { background: '#2196F3', border: '#1976D2', highlight: { background: '#42A5F5', border: '#1976D2' } };
        if (typeLower.includes('linear') || typeLower.includes('fc')) 
            return { background: '#4CAF50', border: '#388E3C', highlight: { background: '#66BB6A', border: '#388E3C' } };
        if (typeLower.includes('relu') || typeLower.includes('activation')) 
            return { background: '#E91E63', border: '#C2185B', highlight: { background: '#EC407A', border: '#C2185B' } };
        if (typeLower.includes('dropout')) 
            return { background: '#9C27B0', border: '#7B1FA2', highlight: { background: '#AB47BC', border: '#7B1FA2' } };
        if (typeLower.includes('batchnorm')) 
            return { background: '#00BCD4', border: '#0097A7', highlight: { background: '#26C6DA', border: '#0097A7' } };
        if (typeLower.includes('input')) 
            return { background: '#FF5722', border: '#D84315', highlight: { background: '#FF7043', border: '#D84315' } };
        if (typeLower.includes('output')) 
            return { background: '#607D8B', border: '#455A64', highlight: { background: '#78909C', border: '#455A64' } };
        
        return { background: '#9E9E9E', border: '#616161', highlight: { background: '#BDBDBD', border: '#616161' } };
    }
    
    showError(message) {
        this.container.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--error-color); flex-direction: column; gap: 1rem;">
                <div style="font-size: 2rem;">❌</div>
                <div style="font-size: 1rem; text-align: center;">${message}</div>
            </div>
        `;
    }
    
    toggleFullscreen() {
        this.isFullscreen = !this.isFullscreen;
        
        if (this.isFullscreen) {
            this.container.style.position = 'fixed';
            this.container.style.top = '0';
            this.container.style.left = '0';
            this.container.style.width = '100vw';
            this.container.style.height = '100vh';
            this.container.style.zIndex = '9999';
            this.container.style.background = 'var(--bg-primary)';
        } else {
            this.container.style.position = 'relative';
            this.container.style.top = 'auto';
            this.container.style.left = 'auto';
            this.container.style.width = '100%';
            this.container.style.height = '600px';
            this.container.style.zIndex = 'auto';
        }
        
        // 重新调整网络图
        if (this.network) {
            setTimeout(() => {
                this.network.redraw();
                this.network.fit();
            }, 100);
        }
    }
    
    exportImage() {
        if (!this.network) {
            console.warn('没有可导出的网络图');
            return;
        }
        
        try {
            const canvas = this.network.getCanvas();
            const dataURL = canvas.toDataURL('image/png');
            
            const link = document.createElement('a');
            link.download = `netview_model_${new Date().toISOString().split('T')[0]}.png`;
            link.href = dataURL;
            link.click();
            
            console.log('模型图已导出');
        } catch (error) {
            console.error('导出失败:', error);
        }
    }
}

// 主应用程序
document.addEventListener('DOMContentLoaded', () => {
    // 初始化组件
    const appState = new AppState();
    const notifications = new NotificationManager();
    const graphManager = new EnhancedNetworkGraphManager();
    
    // 获取DOM元素
    const modelCodeEl = document.getElementById('model-code');
    const analyzeButton = document.getElementById('analyze-button');
    const loadingIndicator = document.getElementById('loading-indicator');
    const themeToggle = document.getElementById('theme-toggle');
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    const exportBtn = document.getElementById('export-btn');
    const totalParamsEl = document.getElementById('total-params');
    const totalLayersEl = document.getElementById('total-layers');
    const trainableParamsEl = document.getElementById('trainable-params');
    const modelSizeEl = document.getElementById('model-size');
    const flopsEl = document.getElementById('flops');
    const additionalStats = document.getElementById('additional-stats');
    const modelDetails = document.getElementById('model-details');
    
    const API_BASE_URL = 'http://localhost:8001/api/v1';
    
    // 事件监听器
    themeToggle?.addEventListener('click', () => {
        appState.toggleTheme();
        notifications.show('主题已切换', 'success', 2000);
    });
    
    fullscreenBtn?.addEventListener('click', () => {
        graphManager.toggleFullscreen();
        fullscreenBtn.textContent = graphManager.isFullscreen ? '🔙 退出全屏' : '📱 全屏';
    });
    
    exportBtn?.addEventListener('click', () => {
        graphManager.exportImage();
        notifications.show('图像已导出', 'success', 2000);
    });
    
    // 测试后端连接
    async function testBackendConnection() {
        try {
            console.log('测试后端连接...');
            const response = await fetch(`${API_BASE_URL}/test`);
            if (response.ok) {
                const data = await response.json();
                console.log('后端连接成功:', data);
                notifications.show('后端服务连接正常', 'success', 2000);
                return true;
            } else {
                console.error('后端连接失败:', response.status, response.statusText);
                return false;
            }
        } catch (error) {
            console.error('后端连接异常:', error);
            return false;
        }
    }
    
    // 页面加载时测试连接
    testBackendConnection().then(success => {
        if (!success) {
            notifications.show('警告：无法连接到后端服务', 'warning', 5000);
        }
    });
    
    // 分析按钮点击事件
    analyzeButton?.addEventListener('click', async () => {
        const modelCode = modelCodeEl?.value.trim();
        if (!modelCode) {
            notifications.show('请输入模型代码！', 'error');
            return;
        }
        
        // 重置状态
        appState.setAnalyzing(true);
        appState.updateProgress(0, '准备分析');
        
        // 清空旧数据
        graphManager.destroy();
        resetStatistics();
        showLoading(true);
        
        try {
            console.log('开始模型分析...');
            
            // 1. 提交分析任务
            appState.updateProgress(10, '提交分析任务');
            const analyzeResponse = await fetch(`${API_BASE_URL}/models/analyze`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    code: modelCode,
                    model_name: "UserSubmittedModel",
                    analysis_options: {}
                })
            });
            
            if (!analyzeResponse.ok) {
                const errorText = await analyzeResponse.text();
                throw new Error(`分析请求失败 (${analyzeResponse.status}): ${errorText}`);
            }
            
            const analyzeResult = await analyzeResponse.json();
            const taskId = analyzeResult.task_id;
            console.log('任务ID:', taskId);
            
            if (!taskId) {
                throw new Error('未能获取任务ID');
            }
            
            // 2. 轮询任务状态
            let taskStatus = '';
            let attempts = 0;
            const maxAttempts = 30;
            const pollInterval = 2000;
            
            while (taskStatus !== 'completed' && attempts < maxAttempts) {
                attempts++;
                await new Promise(resolve => setTimeout(resolve, pollInterval));
                
                const progress = Math.min(90, 20 + (attempts / maxAttempts) * 60);
                appState.updateProgress(progress, '分析模型结构');
                
                const statusResponse = await fetch(`${API_BASE_URL}/models/${taskId}/status`);
                if (!statusResponse.ok) {
                    throw new Error(`任务状态查询失败: ${statusResponse.statusText}`);
                }
                
                const statusResult = await statusResponse.json();
                taskStatus = statusResult.status;
                
                if (statusResult.progress) {
                    appState.updateProgress(statusResult.progress, statusResult.current_stage || '处理中');
                }
                
                if (taskStatus === 'failed') {
                    throw new Error(`模型分析失败: ${statusResult.error || '未知错误'}`);
                }
            }
            
            if (taskStatus !== 'completed') {
                throw new Error('分析任务超时');
            }
            
            // 3. 获取可视化数据
            appState.updateProgress(95, '生成可视化');
            const vizResponse = await fetch(`${API_BASE_URL}/models/${taskId}/visualization`);
            if (!vizResponse.ok) {
                const errorText = await vizResponse.text();
                throw new Error(`获取可视化数据失败 (${vizResponse.status}): ${errorText}`);
            }
            
            const vizData = await vizResponse.json();
            
            // 4. 渲染结果
            appState.updateProgress(100, '完成');
            graphManager.render(vizData.nodes, vizData.edges);
            updateStatistics(vizData);
            
            notifications.show('模型分析完成！', 'success');
            console.log('分析完成');
            
        } catch (error) {
            console.error('分析失败:', error);
            notifications.show(`分析失败: ${error.message}`, 'error', 5000);
            graphManager.showError(error.message);
        } finally {
            appState.setAnalyzing(false);
            showLoading(false);
        }
    });
    
    // 辅助函数
    function showLoading(show) {
        if (loadingIndicator) {
            loadingIndicator.classList.toggle('hidden', !show);
        }
    }
    
    function resetStatistics() {
        if (totalParamsEl) totalParamsEl.textContent = '-';
        if (totalLayersEl) totalLayersEl.textContent = '-';
        if (trainableParamsEl) trainableParamsEl.textContent = '-';
        if (modelSizeEl) modelSizeEl.textContent = '-';
        if (flopsEl) flopsEl.textContent = '-';
        
        if (additionalStats) additionalStats.classList.add('hidden');
        if (modelDetails) modelDetails.classList.add('hidden');
    }
    
    function updateStatistics(vizData) {
        if (vizData.metadata && vizData.metadata.complexity) {
            const complexity = vizData.metadata.complexity;
            
            if (totalParamsEl && complexity.total_parameters) {
                totalParamsEl.textContent = complexity.total_parameters.toLocaleString();
            }
            if (totalLayersEl && complexity.total_layers) {
                totalLayersEl.textContent = complexity.total_layers.toLocaleString();
            }
            if (trainableParamsEl && complexity.trainable_parameters) {
                trainableParamsEl.textContent = complexity.trainable_parameters.toLocaleString();
            }
            
            // 显示额外统计信息
            if (complexity.model_size_mb && modelSizeEl) {
                modelSizeEl.textContent = `${complexity.model_size_mb.toFixed(2)} MB`;
                additionalStats?.classList.remove('hidden');
            }
            
            if (complexity.flops && flopsEl) {
                flopsEl.textContent = formatFlops(complexity.flops);
                additionalStats?.classList.remove('hidden');
            }
            
        } else if (vizData.statistics) {
            // 备用统计信息
            if (totalParamsEl && vizData.statistics.total_parameters) {
                totalParamsEl.textContent = vizData.statistics.total_parameters.toLocaleString();
            }
            if (totalLayersEl && vizData.statistics.node_count) {
                totalLayersEl.textContent = vizData.statistics.node_count.toLocaleString();
            }
        }
    }
    
    function formatFlops(flops) {
        if (flops >= 1e12) {
            return `${(flops / 1e12).toFixed(2)} TFLOPs`;
        } else if (flops >= 1e9) {
            return `${(flops / 1e9).toFixed(2)} GFLOPs`;
        } else if (flops >= 1e6) {
            return `${(flops / 1e6).toFixed(2)} MFLOPs`;
        } else if (flops >= 1e3) {
            return `${(flops / 1e3).toFixed(2)} KFLOPs`;
        }
        return `${flops} FLOPs`;
    }
    
    // 键盘快捷键
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            switch (e.key) {
                case 'Enter':
                    e.preventDefault();
                    if (!appState.isAnalyzing) {
                        analyzeButton?.click();
                    }
                    break;
                case 'd':
                    e.preventDefault();
                    appState.toggleTheme();
                    break;
                case 's':
                    e.preventDefault();
                    graphManager.exportImage();
                    break;
            }
        }
        
        if (e.key === 'F11') {
            e.preventDefault();
            graphManager.toggleFullscreen();
        }
    });
    
    console.log('NetView 应用初始化完成');
    notifications.show('NetView 已准备就绪', 'success', 2000);
});

// 添加CSS动画
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
