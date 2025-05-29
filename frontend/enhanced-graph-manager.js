/**
 * 增强型网络图管理器
 * 专注于改进模型结构图的可视化效果
 */

class EnhancedNetworkGraphManager {
    constructor() {
        this.network = null;
        this.container = document.getElementById('network-graph');
        this.isFullscreen = false;
        this.currentLayout = 'hierarchical';
        this.selectedNodes = new Set();
        this.filterSettings = {
            showLayerTypes: new Set(['all']),
            showConnections: new Set(['all']),
            hideInactiveLayers: false
        };
        
        // 定义增强的节点样式配置
        this.nodeStyleConfig = {
            'conv': {
                shape: 'box',
                color: { background: '#FF6B6B', border: '#FF5252', highlight: { background: '#FF8A80', border: '#FF5252' } },
                icon: '🔲',
                category: 'feature_extraction'
            },
            'linear': {
                shape: 'ellipse',
                color: { background: '#4ECDC4', border: '#26A69A', highlight: { background: '#80CBC4', border: '#26A69A' } },
                icon: '⚫',
                category: 'classifier'
            },
            'pool': {
                shape: 'triangle',
                color: { background: '#45B7D1', border: '#2196F3', highlight: { background: '#81C784', border: '#2196F3' } },
                icon: '🔽',
                category: 'pooling'
            },
            'activation': {
                shape: 'diamond',
                color: { background: '#96CEB4', border: '#66BB6A', highlight: { background: '#A5D6A7', border: '#66BB6A' } },
                icon: '⚡',
                category: 'activation'
            },
            'norm': {
                shape: 'square',
                color: { background: '#FFEAA7', border: '#FFC107', highlight: { background: '#FFE082', border: '#FFC107' } },
                icon: '📊',
                category: 'normalization'
            },
            'dropout': {
                shape: 'dot',
                color: { background: '#DDA0DD', border: '#9C27B0', highlight: { background: '#CE93D8', border: '#9C27B0' } },
                icon: '❌',
                category: 'regularization'
            },
            'attention': {
                shape: 'star',
                color: { background: '#FFB74D', border: '#FF9800', highlight: { background: '#FFCC02', border: '#FF9800' } },
                icon: '👁️',
                category: 'attention'
            },
            'embedding': {
                shape: 'box',
                color: { background: '#F8BBD9', border: '#E91E63', highlight: { background: '#F48FB1', border: '#E91E63' } },
                icon: '📝',
                category: 'embedding'
            }
        };
        
        // 连接类型样式配置
        this.edgeStyleConfig = {
            'sequential': {
                color: '#666666',
                width: 2,
                dashes: false,
                label: '',
                arrows: 'to'
            },
            'residual': {
                color: '#FF5722',
                width: 3,
                dashes: [5, 5],
                label: '+',
                arrows: 'to'
            },
            'attention': {
                color: '#FF9800',
                width: 2,
                dashes: [10, 5],
                label: '👁️',
                arrows: 'to'
            },
            'branch': {
                color: '#2196F3',
                width: 2,
                dashes: [15, 5, 5, 5],
                label: '⑂',
                arrows: 'to'
            },
            'merge': {
                color: '#4CAF50',
                width: 3,
                dashes: false,
                label: '⊕',
                arrows: 'to'
            },
            'dense': {
                color: '#9C27B0',
                width: 2,
                dashes: [2, 3],
                label: '🔗',
                arrows: 'to'
            }
        };
        
        this.layoutConfigs = {
            'hierarchical': {
                layout: {
                    hierarchical: {
                        direction: "UD",
                        sortMethod: "directed",
                        levelSeparation: 120,
                        nodeSpacing: 100,
                        treeSpacing: 200,
                        blockShifting: true,
                        edgeMinimization: true,
                        parentCentralization: true
                    }
                },
                physics: { enabled: false }
            },
            'network': {
                layout: {
                    randomSeed: 2
                },
                physics: {
                    enabled: true,
                    stabilization: { iterations: 100 },
                    barnesHut: {
                        gravitationalConstant: -2000,
                        centralGravity: 0.3,
                        springLength: 95,
                        springConstant: 0.04,
                        damping: 0.09
                    }
                }
            },
            'circular': {
                layout: {
                    randomSeed: 3,
                    improvedLayout: true
                },
                physics: {
                    enabled: true,
                    stabilization: { iterations: 200 },
                    repulsion: {
                        centralGravity: 0.2,
                        springLength: 200,
                        springConstant: 0.05,
                        nodeDistance: 100,
                        damping: 0.09
                    }
                }
            }
        };
    }
    
    destroy() {
        if (this.network) {
            this.network.destroy();
            this.network = null;
        }
    }
    
    render(nodesData, edgesData, metadata = {}) {
        if (!nodesData || !edgesData) {
            this.showError('无法加载模型图数据');
            return;
        }

        console.log('🎨 使用增强渲染器绘制网络图');
        console.log(`📊 节点数: ${nodesData.length}, 边数: ${edgesData.length}`);

        // 清空旧图
        this.destroy();
        this.container.innerHTML = '';

        // 增强节点处理
        const visNodes = this.processEnhancedNodes(nodesData, metadata);
        
        // 增强边处理
        const visEdges = this.processEnhancedEdges(edgesData, metadata);

        const data = {
            nodes: new vis.DataSet(visNodes),
            edges: new vis.DataSet(visEdges),
        };

        const options = this.getLayoutOptions();

        try {
            this.network = new vis.Network(this.container, data, options);
            
            // 添加增强事件监听
            this.setupEnhancedEventListeners();
            
            console.log('✅ 增强网络图渲染完成');
            
            // 创建控制面板
            this.createControlPanel();
            
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
            console.error('❌ 增强网络图渲染失败:', error);
            this.showError('网络图渲染失败: ' + error.message);
        }
    }
    
    processEnhancedNodes(nodesData, metadata) {
        return nodesData.map(node => {
            const nodeType = this.categorizeNodeType(node.type);
            const style = this.nodeStyleConfig[nodeType] || this.nodeStyleConfig['linear'];
            
            return {
                id: node.id,
                label: this.formatEnhancedNodeLabel(node),
                title: this.formatEnhancedNodeTooltip(node, metadata),
                shape: style.shape,
                color: style.color,
                font: {
                    size: 12,
                    face: 'Arial',
                    color: '#333333',
                    strokeWidth: 1,
                    strokeColor: '#ffffff'
                },
                borderWidth: 2,
                margin: 12,
                size: this.calculateNodeSize(node),
                // 添加自定义属性
                nodeType: nodeType,
                category: style.category,
                icon: style.icon,
                parameters: node.num_parameters || 0,
                inputShape: node.input_shape,
                outputShape: node.output_shape,
                flops: node.flops || 0,
                memoryUsage: node.memory_usage || 0
            };
        });
    }
    
    processEnhancedEdges(edgesData, metadata) {
        return edgesData.map((edge, index) => {
            const connectionType = edge.connection_type || 'sequential';
            const style = this.edgeStyleConfig[connectionType] || this.edgeStyleConfig['sequential'];
            
            return {
                id: `edge_${index}`,
                from: edge.source,
                to: edge.target,
                arrows: style.arrows,
                label: style.label,
                color: {
                    color: style.color,
                    highlight: this.lightenColor(style.color, 20),
                    hover: this.lightenColor(style.color, 10)
                },
                width: style.width,
                dashes: style.dashes,
                smooth: this.getEdgeSmoothing(connectionType, index),
                font: {
                    size: 10,
                    color: style.color,
                    strokeWidth: 2,
                    strokeColor: '#ffffff',
                    align: 'middle'
                },
                // 添加自定义属性
                connectionType: connectionType,
                dataFlow: edge.data_flow || {},
                tensorShape: edge.tensor_shape,
                bandwidth: edge.bandwidth || 1
            };
        });
    }
    
    categorizeNodeType(type) {
        const typeLower = type.toLowerCase();
        
        if (typeLower.includes('conv')) return 'conv';
        if (typeLower.includes('linear') || typeLower.includes('dense') || typeLower.includes('fc')) return 'linear';
        if (typeLower.includes('pool')) return 'pool';
        if (typeLower.includes('relu') || typeLower.includes('activation') || typeLower.includes('sigmoid') || typeLower.includes('tanh')) return 'activation';
        if (typeLower.includes('batch') || typeLower.includes('layer') || typeLower.includes('norm')) return 'norm';
        if (typeLower.includes('dropout')) return 'dropout';
        if (typeLower.includes('attention') || typeLower.includes('self_attention')) return 'attention';
        if (typeLower.includes('embedding')) return 'embedding';
        
        return 'linear'; // 默认类型
    }
    
    formatEnhancedNodeLabel(node) {
        const type = node.type || 'Unknown';
        const params = node.num_parameters || 0;
        const shape = this.formatShape(node.output_shape);
        
        let label = type;
        
        // 添加参数信息
        if (params > 0) {
            label += `\n${this.formatNumber(params)}`;
        }
        
        // 添加形状信息
        if (shape && shape !== 'Dynamic') {
            label += `\n[${shape}]`;
        }
        
        return label;
    }
    
    formatEnhancedNodeTooltip(node, metadata) {
        let tooltip = `<div style="font-family: Arial; max-width: 300px;">`;
        
        // 基本信息
        tooltip += `<h4 style="margin: 0 0 8px 0; color: #2196F3;">${node.type}</h4>`;
        tooltip += `<p><strong>ID:</strong> ${node.id}</p>`;
        
        // 参数信息
        if (node.num_parameters) {
            tooltip += `<p><strong>参数数量:</strong> ${this.formatNumber(node.num_parameters)}</p>`;
        }
        
        // 形状信息
        if (node.input_shape) {
            tooltip += `<p><strong>输入形状:</strong> ${this.formatShape(node.input_shape)}</p>`;
        }
        if (node.output_shape) {
            tooltip += `<p><strong>输出形状:</strong> ${this.formatShape(node.output_shape)}</p>`;
        }
        
        // 计算信息
        if (node.flops) {
            tooltip += `<p><strong>FLOPs:</strong> ${this.formatFlops(node.flops)}</p>`;
        }
        
        if (node.memory_usage) {
            tooltip += `<p><strong>内存占用:</strong> ${this.formatMemory(node.memory_usage)}</p>`;
        }
        
        // 配置参数
        if (node.parameters && typeof node.parameters === 'object') {
            tooltip += `<hr style="margin: 8px 0;">`;
            tooltip += `<h5 style="margin: 4px 0;">配置参数:</h5>`;
            for (const [key, value] of Object.entries(node.parameters)) {
                if (value !== null && value !== undefined) {
                    tooltip += `<p style="margin: 2px 0;"><em>${key}:</em> ${value}</p>`;
                }
            }
        }
        
        tooltip += `</div>`;
        return tooltip;
    }
    
    calculateNodeSize(node) {
        const baseSize = 20;
        const params = node.num_parameters || 0;
        
        if (params === 0) return baseSize;
        
        // 基于参数数量的对数scale调整大小
        const logParams = Math.log10(params + 1);
        return Math.min(baseSize + logParams * 5, 50);
    }
    
    getEdgeSmoothing(connectionType, index) {
        const smoothingConfigs = {
            'sequential': {
                type: 'cubicBezier',
                forceDirection: 'vertical',
                roundness: 0.1
            },
            'residual': {
                type: 'curvedCW',
                roundness: 0.3 + (index % 3) * 0.1
            },
            'attention': {
                type: 'curvedCCW',
                roundness: 0.2
            },
            'branch': {
                type: 'dynamic',
                roundness: 0.4
            },
            'merge': {
                type: 'continuous',
                roundness: 0.2
            }
        };
        
        return smoothingConfigs[connectionType] || smoothingConfigs['sequential'];
    }
    
    setupEnhancedEventListeners() {
        // 节点选择事件
        this.network.on('selectNode', (params) => {
            this.handleNodeSelection(params);
        });
        
        // 节点悬停事件
        this.network.on('hoverNode', (params) => {
            this.handleNodeHover(params, true);
        });
        
        this.network.on('blurNode', (params) => {
            this.handleNodeHover(params, false);
        });
        
        // 边选择事件
        this.network.on('selectEdge', (params) => {
            this.handleEdgeSelection(params);
        });
        
        // 双击事件
        this.network.on('doubleClick', (params) => {
            this.handleDoubleClick(params);
        });
        
        // 右键菜单
        this.network.on('oncontext', (params) => {
            this.handleRightClick(params);
        });
    }
    
    handleNodeSelection(params) {
        const nodeIds = params.nodes;
        console.log('🎯 选中节点:', nodeIds);
        
        if (nodeIds.length > 0) {
            this.selectedNodes = new Set(nodeIds);
            this.highlightRelatedNodes(nodeIds[0]);
            this.showNodeDetails(nodeIds[0]);
        } else {
            this.selectedNodes.clear();
            this.clearHighlights();
            this.hideNodeDetails();
        }
    }
    
    handleNodeHover(params, isHovering) {
        const nodeId = params.node;
        if (isHovering && nodeId) {
            this.container.style.cursor = 'pointer';
            // 可以添加更多悬停效果
        } else {
            this.container.style.cursor = 'default';
        }
    }
    
    handleEdgeSelection(params) {
        const edgeIds = params.edges;
        console.log('🔗 选中边:', edgeIds);
        
        if (edgeIds.length > 0) {
            this.showEdgeDetails(edgeIds[0]);
        }
    }
    
    handleDoubleClick(params) {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            this.focusOnNode(nodeId);
        }
    }
    
    handleRightClick(params) {
        params.event.preventDefault();
        // 这里可以添加右键菜单逻辑
        console.log('🖱️ 右键点击:', params);
    }
    
    createControlPanel() {
        // 创建控制面板HTML
        const controlPanelHTML = `
            <div id="graph-controls" style="position: absolute; top: 10px; right: 10px; background: var(--bg-primary); padding: 10px; border-radius: 8px; box-shadow: var(--shadow-md); z-index: 100;">
                <div style="margin-bottom: 8px;">
                    <label for="layout-select" style="font-size: 12px; color: var(--text-secondary);">布局:</label>
                    <select id="layout-select" style="margin-left: 5px; font-size: 11px;">
                        <option value="hierarchical">分层布局</option>
                        <option value="network">网络布局</option>
                        <option value="circular">环形布局</option>
                    </select>
                </div>
                <div style="margin-bottom: 8px;">
                    <button id="filter-btn" style="font-size: 11px; padding: 4px 8px;">🔍 筛选</button>
                    <button id="layout-btn" style="font-size: 11px; padding: 4px 8px;">🎨 重新布局</button>
                </div>
                <div>
                    <button id="fit-btn" style="font-size: 11px; padding: 4px 8px;">📐 适应窗口</button>
                    <button id="reset-btn" style="font-size: 11px; padding: 4px 8px;">🔄 重置</button>
                </div>
            </div>
        `;
        
        // 将控制面板添加到容器
        this.container.insertAdjacentHTML('afterbegin', controlPanelHTML);
        
        // 绑定事件
        this.bindControlEvents();
    }
    
    bindControlEvents() {
        const layoutSelect = document.getElementById('layout-select');
        const filterBtn = document.getElementById('filter-btn');
        const layoutBtn = document.getElementById('layout-btn');
        const fitBtn = document.getElementById('fit-btn');
        const resetBtn = document.getElementById('reset-btn');
        
        layoutSelect?.addEventListener('change', (e) => {
            this.changeLayout(e.target.value);
        });
        
        filterBtn?.addEventListener('click', () => {
            this.showFilterDialog();
        });
        
        layoutBtn?.addEventListener('click', () => {
            this.redrawGraph();
        });
        
        fitBtn?.addEventListener('click', () => {
            this.network.fit({ animation: true });
        });
        
        resetBtn?.addEventListener('click', () => {
            this.resetGraph();
        });
    }
    
    changeLayout(layoutType) {
        this.currentLayout = layoutType;
        const options = this.getLayoutOptions();
        this.network.setOptions(options);
        console.log(`🔄 切换到${layoutType}布局`);
    }
    
    getLayoutOptions() {
        const baseOptions = {
            interaction: {
                dragNodes: true,
                dragView: true,
                zoomView: true,
                hover: true,
                selectConnectedEdges: true,
                tooltipDelay: 300,
                hideEdgesOnDrag: false,
                hideNodesOnDrag: false
            },
            nodes: {
                font: {
                    size: 12,
                    face: 'Arial',
                    color: '#333333',
                    strokeWidth: 1,
                    strokeColor: '#ffffff'
                },
                margin: 12,
                borderWidth: 2,
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.1)',
                    size: 8,
                    x: 3,
                    y: 3
                }
            },
            edges: {
                font: {
                    size: 10,
                    align: 'middle',
                    strokeWidth: 2,
                    strokeColor: '#ffffff'
                },
                smooth: true,
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.05)',
                    size: 4,
                    x: 2,
                    y: 2
                }
            }
        };
        
        return Object.assign({}, baseOptions, this.layoutConfigs[this.currentLayout]);
    }
    
    // 工具函数
    formatNumber(num) {
        if (num >= 1000000000) {
            return (num / 1000000000).toFixed(1) + 'B';
        } else if (num >= 1000000) {
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
    
    formatFlops(flops) {
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
    
    formatMemory(bytes) {
        if (bytes >= 1024 * 1024 * 1024) {
            return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
        } else if (bytes >= 1024 * 1024) {
            return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
        } else if (bytes >= 1024) {
            return `${(bytes / 1024).toFixed(2)} KB`;
        }
        return `${bytes} B`;
    }
    
    lightenColor(color, percent) {
        const num = parseInt(color.replace("#", ""), 16);
        const amt = Math.round(2.55 * percent);
        const R = (num >> 16) + amt;
        const G = (num >> 8 & 0x00FF) + amt;
        const B = (num & 0x0000FF) + amt;
        return "#" + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 + 
                     (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 + 
                     (B < 255 ? B < 1 ? 0 : B : 255)).toString(16).slice(1);
    }
    
    showError(message) {
        this.container.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--error-color); flex-direction: column; gap: 1rem;">
                <div style="font-size: 2rem;">❌</div>
                <div style="font-size: 1rem; text-align: center;">${message}</div>
                <button onclick="location.reload()" style="margin-top: 1rem; padding: 0.5rem 1rem; background: var(--primary-color); color: white; border: none; border-radius: 4px; cursor: pointer;">重新加载</button>
            </div>
        `;
    }
    
    // 占位符方法，后续实现
    highlightRelatedNodes(nodeId) {
        console.log('🎯 高亮相关节点:', nodeId);
    }
    
    clearHighlights() {
        console.log('🔄 清除高亮');
    }
    
    showNodeDetails(nodeId) {
        console.log('📋 显示节点详情:', nodeId);
    }
    
    hideNodeDetails() {
        console.log('❌ 隐藏节点详情');
    }
    
    showEdgeDetails(edgeId) {
        console.log('🔗 显示边详情:', edgeId);
    }
    
    focusOnNode(nodeId) {
        console.log('🎯 聚焦节点:', nodeId);
        this.network.focus(nodeId, {
            animation: {
                duration: 1000,
                easingFunction: 'easeInOutQuad'
            }
        });
    }
    
    showFilterDialog() {
        console.log('🔍 显示筛选对话框');
    }
    
    redrawGraph() {
        console.log('🎨 重新绘制图形');
        this.network.redraw();
    }
    
    resetGraph() {
        console.log('🔄 重置图形');
        this.selectedNodes.clear();
        this.clearHighlights();
        this.network.fit({ animation: true });
    }
    
    // 兼容性方法
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
            link.download = `netview_enhanced_model_${new Date().toISOString().split('T')[0]}.png`;
            link.href = dataURL;
            link.click();
            
            console.log('📸 增强模型图已导出');
        } catch (error) {
            console.error('导出失败:', error);
        }
    }
}

// 导出类以供使用
window.EnhancedNetworkGraphManager = EnhancedNetworkGraphManager;
