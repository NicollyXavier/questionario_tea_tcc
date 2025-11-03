async function calcularRisco() {
    const alerta_corretos = {
        "P1": 1,
        "P2": 0,
        "P3": 0,
        "P4": 0,
        "P5": 0,
        "P6": 0,
        "P7": 0,
        "P8": 1,
        "P9": 0,
        "P10": 0,
        "P11": 0,
        "P12": 1
    };

    let respostas = [];
    let respostasAlerta = 0;

    // Coletar respostas e contar alertas
    for (let p in alerta_corretos) {
        let radios = document.getElementsByName(p);
        for (let i = 0; i < radios.length; i++) {
            if (radios[i].checked) {
                const valor = parseInt(radios[i].value);
                respostas.push(valor);
                if (valor === alerta_corretos[p]) {
                    respostasAlerta++;
                }
            }
        }
    }

    // Cálculo local simples (backup)
    let resultadoLocal = '';
    if (respostasAlerta <= 2) {
        resultadoLocal = 'Risco baixo';
    } else if (respostasAlerta <= 5) {
        resultadoLocal = 'Risco moderado';
    } else {
        resultadoLocal = 'Alto risco';
    }

    document.getElementById('resultado').innerText = 'Avaliando com IA... 🔄';

    // Enviar para o servidor Flask
    try {
        const response = await fetch("https://mchat-backend-bg0s.onrender.com/avaliar", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ respostas })
        });

        const data = await response.json();

        if (data.risco) {
            document.getElementById('resultado').innerText = `🤖 Resultado com IA: RISCO ${data.risco.toUpperCase()}`;
        } else {
            document.getElementById('resultado').innerText = `⚠️ Erro na IA: ${data.erro || "Resposta inválida"}`;
        }

    } catch (error) {
        console.error(error);
        document.getElementById('resultado').innerText = `⚠️ Erro ao conectar à IA. Mostrando cálculo local: ${resultadoLocal}`;
    }
}

