// 等待DOM加载完成
document.addEventListener('DOMContentLoaded', function() {
    // 添加打字特效
    const titleElement = document.querySelector('.hero-content h1');
    const subtitleElement = document.querySelector('.hero-content h2');
    const originalTitle = titleElement.textContent;
    const originalSubtitle = subtitleElement.textContent;
    
    // 清空标题内容，准备添加打字效果
    titleElement.textContent = '';
    subtitleElement.textContent = '';
    
    // 为主标题添加打字效果
    let titleIndex = 0;
    function typeTitle() {
        if (titleIndex < originalTitle.length) {
            titleElement.textContent += originalTitle.charAt(titleIndex);
            titleIndex++;
            setTimeout(typeTitle, 150);
        } else {
            // 主标题完成后，开始副标题
            setTimeout(typeSubtitle, 500);
        }
    }
    
    // 为副标题添加打字效果
    let subtitleIndex = 0;
    function typeSubtitle() {
        if (subtitleIndex < originalSubtitle.length) {
            subtitleElement.textContent += originalSubtitle.charAt(subtitleIndex);
            subtitleIndex++;
            setTimeout(typeSubtitle, 50);
        }
    }
    
    // 开始打字效果
    setTimeout(typeTitle, 500);

    // 导航栏滚动效果
    const nav = document.querySelector('nav');
    const navHeight = nav.offsetHeight;
    
    window.addEventListener('scroll', function() {
        if (window.scrollY > 100) {
            nav.classList.add('scrolled');
        } else {
            nav.classList.remove('scrolled');
        }
    });

    // 增强所有按钮的点击效果
    const allButtons = document.querySelectorAll('.btn');
    allButtons.forEach(button => {
        button.addEventListener('mousedown', function() {
            this.style.transform = 'scale(0.95)';
        });
        
        button.addEventListener('mouseup', function() {
            this.style.transform = '';
        });
        
        button.addEventListener('mouseleave', function() {
            this.style.transform = '';
        });
    });

    // 平滑滚动到锚点
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                const navHeight = nav.offsetHeight;
                const targetPosition = targetElement.getBoundingClientRect().top + window.pageYOffset;
                
                // 添加点击反馈
                if (this.classList.contains('btn')) {
                    this.classList.add('btn-clicked');
                    setTimeout(() => {
                        this.classList.remove('btn-clicked');
                    }, 300);
                }
                
                window.scrollTo({
                    top: targetPosition - navHeight,
                    behavior: 'smooth'
                });
                
                // 更新URL哈希，但不触发滚动
                history.pushState(null, null, targetId);
                
                // 添加高亮效果到目标部分
                const sections = document.querySelectorAll('section');
                sections.forEach(section => section.classList.remove('highlight-section'));
                targetElement.classList.add('highlight-section');
                setTimeout(() => {
                    targetElement.classList.remove('highlight-section');
                }, 1000);
            }
        });
    });

    // 特别处理"Get Started"和"See Demo"按钮
    const getStartedBtn = document.querySelector('a.btn.primary[href="#setup"]');
    const seeDemoBtn = document.querySelector('a.btn.secondary[href="#demo"]');
    
    if (getStartedBtn) {
        getStartedBtn.addEventListener('click', function(e) {
            // 已经在通用处理程序中处理了滚动，这里可以添加额外效果
            console.log('Get Started button clicked');
        });
    }
    
    if (seeDemoBtn) {
        seeDemoBtn.addEventListener('click', function(e) {
            // 已经在通用处理程序中处理了滚动，这里可以添加额外效果
            console.log('See Demo button clicked');
        });
    }

    // 特性卡片动画
    const featureCards = document.querySelectorAll('.feature-card');
    
    // 使用Intersection Observer检测元素是否在视口中
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: `-${navHeight}px 0px 0px 0px`
    });
    
    featureCards.forEach(card => {
        observer.observe(card);
    });

    // 为代码块添加复制功能
    const codeBlocks = document.querySelectorAll('.code-block');
    
    codeBlocks.forEach(block => {
        const codeText = block.querySelector('code').innerText;
        
        // 创建复制按钮
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-btn';
        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
        copyButton.title = 'Copy code';
        
        // 添加复制功能
        copyButton.addEventListener('click', () => {
            navigator.clipboard.writeText(codeText)
                .then(() => {
                    copyButton.innerHTML = '<i class="fas fa-check"></i>';
                    setTimeout(() => {
                        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                    }, 2000);
                })
                .catch(err => {
                    console.error('Unable to copy text: ', err);
                });
        });
        
        block.appendChild(copyButton);
    });
}); 